import time
import json
import torch
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf


def sqlite_query(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()


def l2_proj(z, o, epsilon):
    noise = z - o
    noise_norm = noise / torch.norm(noise) * epsilon
    return o + noise_norm


projections = dict(
    linf=lambda z, o, epsilon: z.add_(-1, o).clamp_(-epsilon, epsilon).add_(o),
    l2=l2_proj
)


def requires_grad(models, value=False):
    for model in models:
        for p in model.parameters():
            p.requires_grad = value

def to_device(tensors, device):
    moved = []
    for tensor in tensors:
        moved.append(tensor.to(device))
    return moved


def init_opt(ctx):
    cfg = ctx.ex.current_run.config
    opt = dict()
    for k,v in cfg.items():
        opt[k] = v
    return opt


def pgd(x, y, model, criterion, epsilon=0.3, step_size=0.01, num_iterations=100, norm="linf"):
    assert norm in projections.keys()
    model.eval()
    noise = torch.randn(x.shape, device=x.device) * (epsilon)# - epsilon
    Xnoisy = x.data + noise
    Xnoisy = projections[norm](Xnoisy.data, x.data, epsilon)
    Xnoisy.requires_grad = True

    for i in range(num_iterations):
        loss = criterion(model(Xnoisy), y)
        loss.backward()
        Xnoisy.data += step_size*torch.sign(Xnoisy.grad.data)
        Xnoisy.data = projections[norm](Xnoisy.data, x.data, epsilon)
        Xnoisy.grad.data.zero_()
    model.train()
    return Xnoisy.data


def build_filename(ctx, included_opts=('model',), excluded_opts=('root', 'gpu')):
    opt = ctx.opt
    included_opts = set(included_opts)
    cfg_mdf = ctx.ex.current_run.config_modifications.modified
    included_opts = included_opts.union(cfg_mdf)
    o = {k: opt[k] for k in included_opts if k in opt and k not in excluded_opts}
    o['offline'] = True if opt['offline'] else False
    t = time.strftime('%b_%d_%H_%M_%S')
    opt['time'] = t
    opt['filename'] = f"({t})_opts_{json.dumps(o, sort_keys=True, separators=(',', ':'))}"


def save_checkpoint(opt, iteration, stats):
    torch.save(dict(opt=opt, i=iteration, stats=stats),
               join(opt['save_folder'], 'checkpoint.pkl'))


def schedule(optimizer, f: callable):
    for param_group in optimizer.param_groups:
        param_group['lr'] = f(param_group['lr'])


def load_checkpoint(path):
    dict = torch.load(path)
    return dict['opt'], dict['i'], dict['stats']


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class RunningMeanStd:

    def __init__(self):
        self.m = None
        self.count = 0
        self.mean = None
        self.init = False
        self.first = None

    def update(self, x: torch.Tensor):
        if self.first is not None:
            x = torch.cat([self.first, x], dim=0)
            self.first = None

        mean_x, count_x, var_x = x.mean(dim=0), x.shape[0], x.var(dim=0)
        m_x = var_x * (count_x - 1)

        if not self.init:
            if count_x == 1:
                self.first = x
                return
            self.mean = mean_x
            self.count = count_x
            self.m = m_x
            self.init = True
            return

        delta = mean_x - self.mean
        self.m = m_x + self.m + delta**2 * count_x * self.count / (self.count + count_x)
        self.mean += delta*count_x/(self.count + count_x)
        self.count += count_x

    @property
    def std(self):
        return torch.sqrt(self.m/(self.count - 1))


class EpisodeStats:

    def __init__(self, buffer_size=40):
        self.rewards = []
        self.episode_lengths = []
        self.buffer_size = buffer_size

    def update(self, rewards, lengths):
        self.rewards += rewards
        self.episode_lengths += lengths

    def mean(self):
        rewards = np.array(self.rewards)
        lenghts = np.array(self.episode_lengths)
        if len(self.rewards) > self.buffer_size:
            return np.mean(rewards[-self.buffer_size:]), np.mean(lenghts[-self.buffer_size:])
        else:
            return np.mean(rewards), np.mean(lenghts)

    def moving_average(self):
        if len(self.rewards) > self.buffer_size:
            return self._moving_average(np.array(self.rewards), self.buffer_size), \
                   self._moving_average(np.array(self.episode_lengths), self.buffer_size)

    def _moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.not_done[ind])
        )