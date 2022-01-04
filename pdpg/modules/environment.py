import numpy as np
import torch
import pickle


class EnvManager:
    def __init__(self, agent, env, warmup=0, parallel=False):
        if parallel:
            self.agent = pickle.loads(agent)
        else:
            self.agent = agent
        self.env = env
        self.current_state = env.reset()
        self.new = True
        self.cur_ep_ret, self.cur_ep_len = 0, 0
        self.iter = 0
        self.warmup = warmup

    def single_rollout(self, horizon, stochastic=True):
        state = self.current_state
        action = self.env.action_space.sample()
        new = self.new

        ep_returns, ep_lengths = [], []

        # Initialize history arrays
        states = np.array([state for _ in range(horizon + 1)])
        rewards = np.zeros((horizon, 1), 'float32')
        news = np.zeros((horizon + 1, 1), 'int32')
        actions = np.array([action for _ in range(horizon)])

        for i in range(horizon + 1):
            self.iter += 1
            state = torch.Tensor(state)[None, :]

            action = self.agent.act(state, stochastic=stochastic) if self.iter > self.warmup else \
                torch.tensor(self.env.action_space.sample())[None, :]

            if i == horizon:
                states[-1:], news[-1:] = state, int(new)
                return {"states": torch.Tensor(states), "rewards": torch.Tensor(rewards), "new": torch.Tensor(news),
                        "actions": torch.Tensor(actions), "ep_returns": ep_returns, "ep_lengths": ep_lengths}

            states[i], news[i], actions[i] = state, int(new), action

            action = action.numpy().squeeze(1) if action.shape[1] == 1 else action
            state, rew, new, _ = self.env.step(action)
            rewards[i] = rew

            self.cur_ep_ret += rew
            self.cur_ep_len += 1
            if new:
                ep_returns.append(self.cur_ep_ret)
                ep_lengths.append(self.cur_ep_len)
                self.cur_ep_ret = 0
                self.cur_ep_len = 0
                state = self.env.reset()
            self.current_state = state
            self.new = new

