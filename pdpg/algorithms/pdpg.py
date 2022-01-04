import torch
import copy
import torch.nn as nn
from pdpg.modules.utils import ReplayBuffer, pgd, requires_grad
from pdpg.modules.models import Mlp
from copy import deepcopy

config = dict(
    # Max time steps to run environment for
    max_timesteps=1e6,
    # Std of Gaussian exploration noise relative to max action
    exp_noise=0.1,
    # Batch size for both actor and critic
    batch_size=256,
    # Discount factor
    discount_factor=0.99,
    # Target network update rate
    tau=0.005,
    # Std of Noise added to target policy during critic update relative to max action
    action_noise=0.2,
    # Range to clip target policy noise relative to max action
    noise_clip=0.5,
    # Frequency of delayed policy updates
    policy_freq=2,
    lr=3E-4,
    time_steps=2000,
    alpha=1E-2,
    beta=1E-2,
    implicit_steps=15
)


class Actor(Mlp):
    def __init__(self, input_size, output_size, num_hidden, hidden_sizes, max_action, activation='relu'):
        super().__init__(input_size, output_size, num_hidden, hidden_sizes, activation)
        self.max_action = max_action

    def forward(self, x):
        out = super().forward(x)
        return torch.tanh(out) * self.max_action


class Critic(Mlp):
    def __init__(self, input_size, output_size, num_hidden, hidden_sizes, activation='relu'):
        super().__init__(input_size, output_size, num_hidden, hidden_sizes, activation)

    def forward(self, *args):
        assert 3 > len(args) > 0
        if len(args) == 2:
            c = torch.cat(args, 1)
        else:
            c, = args
        return super().forward(c)


class PDPG(nn.Module):

    def __init__(self, state_size, action_size, max_action, device, lr=3E-4, exp_noise=0.1, action_noise=0.2,
                 noise_clip=0.5, tau=0.005, discount_factor=0.99,
                 max_capacity=1E6, policy_freq=2, weight_decay=0.0):
        super().__init__()

        self.device = device
        self.pi = Actor(state_size, action_size, 2, 256, max_action)
        self.target_pi = copy.deepcopy(self.pi)
        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=lr, weight_decay=weight_decay)

        self.q1, self.q2 = Critic(action_size + state_size, 1, 2, 256), Critic(action_size + state_size, 1, 2, 256)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.optimizer_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        self.discount_factor = discount_factor
        self.exp_noise = exp_noise
        self.action_noise = action_noise
        self.noise_clip = noise_clip
        self.tau = tau
        self.policy_freq = policy_freq
        self.buffer = ReplayBuffer(state_size, action_size, int(max_capacity))
        self.max_action = max_action
        self.iter = 0

    def compute_value(self, state):
        return self.value_network(state)

    def act(self, state, stochastic=True, clip=True):
        with torch.no_grad():
            action = self.pi(state.to(self.device))
            if stochastic:
                noise = torch.randn_like(action) * self.exp_noise
                action = action + noise
            if clip:
                action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu()

    def add(self, st, at, rt, st1, new):
        self.buffer.add(st, at, rt, st1, new)

    def compute_q_loss(self, st, at, rt, st1, new, adv=False):
        with torch.no_grad():
            noise = (torch.randn_like(at) * self.action_noise).clamp(-self.noise_clip, self.noise_clip)
            at1 = (self.target_pi(st1) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.target_q1(st1, at1), self.target_q2(st1, at1)
            target_q = torch.min(target_q1, target_q2)
            target_q = rt + self.discount_factor * target_q * (1 - new)

        mse = nn.MSELoss()
        q1, q2 = self.q1(st, at), self.q2(st, at)
        loss_q1, loss_q2 = mse(q1, target_q), mse(q2, target_q)
        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_pi_loss(self, st):
        return -self.q1(st, self.pi(st)).mean()

    def compute_target(self, at, rt, st1, new):
        with torch.no_grad():
            noise = (torch.randn_like(at) * self.action_noise).clamp(-self.noise_clip, self.noise_clip)
            at1 = (self.target_pi(st1) + noise).clamp(-self.max_action, self.max_action)
            target_q = torch.min(self.target_q1(st1, at1), self.target_q2(st1, at1))
            target_q = rt + self.discount_factor * target_q * (1 - new)
        return target_q

    def implicit_loss(self, st, at, target_q, alpha=0.001, beta=1E-2):
        mse = nn.MSELoss()
        huber = nn.SmoothL1Loss()
        at_pred = self.pi(st)
        loss = huber(self.q1(st, at), target_q) + huber(self.q2(st, at), target_q) \
               - 0.5 * (self.target_q1(st, at_pred) + self.target_q2(st, at_pred)).mean() * beta
        loss += alpha * self._mse(self.q1.parameters(), self.target_q1.parameters())
        loss += alpha * self._mse(self.q2.parameters(), self.target_q2.parameters())
        loss += alpha * self._mse(self.pi.parameters(), self.target_pi.parameters())
        return loss

    def synchronize(self):
        self.target_pi = deepcopy(self.pi)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

    def update_parameters(self):
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.pi.parameters(), self.target_pi.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_step(self, batch_size, implicit_steps=10, alpha=0.01, beta=1E-2):
        self.iter += 1
        st, at, rt, st1, new = self.buffer.sample(batch_size)
        st, at, rt, st1, new = st.to(self.device), at.to(self.device), rt.to(self.device), st1.to(
            self.device), new.to(self.device)
        target_q = self.compute_target(at, rt, st1, new)

        for i in range(implicit_steps):
            self.optimizer_q.zero_grad()
            self.optimizer_pi.zero_grad()
            loss = self.implicit_loss(st, at, target_q, alpha, beta)
            loss.backward()
            self.optimizer_pi.step()
            self.optimizer_q.step()

        self.update_parameters()

    def save(self, path):
        torch.save(dict(
            model=self.state_dict(),
            # buffer=self.buffer,
            # iter=self.iter,
            # optimizer_q=self.optimizer_q.state_dict(),
            # optimizer_pi=self.optimizer_pi.state_dict()
            ), path)

    def load(self, path):
        d = torch.load(path)
        self.load_state_dict(d['model'])
        # self.optimizer_pi.load_state_dict(d['optimizer_pi'])
        # self.optimizer_q.load_state_dict(d['optimizer_q'])
        # self.iter = d['iter']
        # self.buffer = d['buffer']

    def _mse(self, parameters, parameters_old):
        dist = 0
        n = 0
        for p, p_old in zip(parameters, parameters_old):
            dist += ((p - p_old) ** 2).sum()
            n += p.numel()
        return dist / n
