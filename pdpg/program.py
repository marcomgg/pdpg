import gym
import torch
import numpy as np
import time
from pdpg.modules.utils import save_checkpoint, load_checkpoint, EpisodeStats
from pdpg.modules import log
from pdpg.algorithms.pdpg import PDPG
from pdpg.modules.environment import EnvManager
from gym.wrappers import Monitor
from pdpg.environments.wrappers import AntWrapper
from os.path import join, exists
from pdpg.modules.utils import schedule
import random


def log_optimizer_parameters(optimizer, name, logger, iter, tag):
    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            state = optimizer.state[p]
            param = state[name]
            t = f"{tag}:{i}"
            logger.histo_summary(t, param.data.clone().cpu().numpy(), iter)


def log_save_stats(ctx, i, dt, agent, stats):
    opt = ctx.opt
    if i != 0:
        print(f"{opt['log_freq']} Iterations completed in {dt:.2f}s")

    if i > opt['batch_size']:
        log_optimizer_parameters(agent.optimizer_pi, 'exp_avg_sq', ctx.train_logger, i, "policy")
        log_optimizer_parameters(agent.optimizer_q, 'exp_avg_sq', ctx.train_logger, i, "value")

    eval_env = gym.make(opt['environment'])
    if opt['environment'] == "Ant-v2":
        eval_env = AntWrapper(eval_env)
    eval_env.seed(opt['seed'] + 100)
    eval_env.action_space.seed(opt['seed'] + 100)
    ep_returns, ep_lengths, actions = evaluate(ctx, agent, eval_env)
    ctx.train_logger.scalar_summary("Episodic return", np.mean(np.array(ep_returns)), i)
    ctx.train_logger.histo_summary("Actions", actions.numpy(), i)

    stats.update(ep_returns, ep_lengths)
    save_checkpoint(opt, i, stats)
    agent.save(join(opt['save_folder'], 'model.pkl'))


def train_off(ctx, agent, env):
    opt = ctx.opt
    time_steps = int(opt["max_timesteps"])
    stats = ctx.stats
    state, done = env.reset(), False
    dt = 0
    for i in range(ctx.iter, time_steps):
        tm = time.time()
        if i % opt['log_freq'] == 0:
            print(f"Time step {i} / {time_steps}")
        ctx.iter = i

        if i < opt['warmup']:
            action = env.action_space.sample()
        else:
            action = agent.act(torch.tensor(state.astype(np.float32))[None, :])

        next_state, reward, done, _ = env.step(action)
        agent.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, done = env.reset(), False

        if i > opt['batch_size']:
            agent.train_step(opt['batch_size'], opt['implicit_steps'], opt['alpha'], opt['beta'])
            schedule(agent.optimizer_pi, lambda lr: opt['lr'] * np.cos(np.pi/2 * i / (time_steps + 200)))
            schedule(agent.optimizer_q, lambda lr: opt['lr'] * np.cos(np.pi / 2 * i / (time_steps + 200)))

        dt += time.time() - tm
        if i % opt['log_freq'] == 0:
            log_save_stats(ctx, i, dt, agent, stats)
            dt = 0


def train_batch(ctx, agent):
    opt = ctx.opt
    time_steps = int(opt["max_timesteps"])
    stats = ctx.stats
    dt = 0
    for i in range(ctx.iter, time_steps):
        tm = time.time()
        if i % opt['log_freq'] == 0:
            print(f"Time step {i} / {time_steps}")
        ctx.iter = i

        agent.train_step(opt['batch_size'], opt['implicit_steps'], opt['alpha'], opt['beta'])

        dt += time.time() - tm

        if i > opt['batch_size']:
            agent.train_step(opt['batch_size'], opt['implicit_steps'], opt['alpha'], opt['beta'])
            schedule(agent.optimizer_pi, lambda lr: opt['lr'] * np.cos(np.pi/2 * i / (time_steps + 200)))
            schedule(agent.optimizer_q, lambda lr: opt['lr'] * np.cos(np.pi / 2 * i / (time_steps + 200)))

        if i % opt['log_freq'] == 0:
            log_save_stats(ctx, i, dt, agent, stats)
            dt = 0


def evaluate(ctx, agent, env):
    opt = ctx.opt
    time_steps = opt["time_steps"]
    ep_lengths, ep_returns, actions = [], [], []
    manager = EnvManager(agent, env)
    n_eval_episodes = opt['n_eval_episodes']

    while len(ep_returns) < n_eval_episodes:
        data = manager.single_rollout(time_steps, stochastic=False)
        ep_returns += data['ep_returns']
        ep_lengths += data['ep_lengths']
        actions.append(data['actions'])

    ep_returns = ep_returns[0:n_eval_episodes]
    ep_lengths = ep_lengths[0:n_eval_episodes]

    actions = torch.cat(actions, dim=0)
    print(f"Average episode return: {np.array(ep_returns).mean():.2f}")

    save_file = join(opt['save_folder'], f"{opt['environment']}_{opt['seed']}.pkl")
    if exists(save_file):
        file_data = torch.load(save_file)
        training_curve = file_data['curve']
        iterations = file_data['iter']
        curve_all = file_data['curve_all']
    else:
        training_curve = []
        iterations = []
        curve_all = []
    curve_all.append(np.array(ep_returns))
    training_curve.append(np.array(ep_returns).mean())
    iterations.append(ctx.iter)
    torch.save(dict(curve=training_curve, iter=iterations, curve_all=curve_all), save_file)

    return ep_returns, ep_lengths, actions[0:ep_lengths[0]]


def run(ctx):
    opt = ctx.opt
    resume, eval = opt['resume'], opt['evaluate']
    ctx.stats = EpisodeStats(40)

    if resume:
        old_opt = opt
        opt, i, stats = load_checkpoint(join(resume, 'checkpoint.pkl'))
        ctx.opt, ctx.iter, ctx.stats = opt, i+1, stats

    ctx.train_logger = log.TfLogger(join(opt['train_folder'], opt['filename']))
    env = gym.make(opt['environment'])

    if opt['environment'] == "Ant-v2":
        env = AntWrapper(env)

    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("****** cudnn.deterministic is set ******")

    print("num threads set to one")
    torch.set_num_threads(1)

    env.action_space.seed(opt['seed'])
    env.seed(opt['seed'])

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    print(f"Action space size: {action_size} \n"
          f"State space size: {state_size} \n"
          f"Warm-up: {opt['warmup']}")

    kwargs = dict(state_size=state_size, action_size=action_size, max_action=max_action,
                  device=ctx.device, lr=opt['lr'], exp_noise=opt['exp_noise'] * max_action,
                  action_noise=opt['action_noise'] * max_action, noise_clip=opt['noise_clip'] * max_action,
                  tau=opt['tau'], discount_factor=opt['discount_factor'], policy_freq=opt['policy_freq'],
                  weight_decay=opt['wd'])
    agent = PDPG(**kwargs)
    agent = agent.to(ctx.device)

    if resume:
        agent.load(join(resume, 'model.pkl'))

    if opt['offline']:
        print("Training offline...")
        d = torch.load(opt['offline'])
        agent.buffer = d['buffer']
        train_batch(ctx, agent)
        return

    if eval:
        print("Evaluating model...")
        opt['n_eval_episodes'] = old_opt['n_eval_episodes']
        env = Monitor(env, opt['save_folder'], force=True,
                      write_upon_reset=True,
                      video_callable=lambda episode_id: True)
        evaluate(ctx, agent, env)
        return

    train_off(ctx, agent, env)
