#!/usr/bin/env python
# coding: utf-8

import gym
import argparse
import numpy as np
import torch
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Buffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

        self.absorbing_state = np.zeros(state_shape, dtype=np.float32)
        self.zero_action = np.zeros(action_shape, dtype=np.float32)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def add(self, data, reward, done):
        state, action, next_state = data
        self.append(state, action, reward, done, next_state)

    def addAbsorbing(self):
        self.append(self.absorbing_state, self.zero_action, 0, False, self.absorbing_state)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)

class Actor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

class Critic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim + action_dim, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, 1)

        self.l4 = torch.nn.Linear(state_dim + action_dim, 400)
        self.l5 = torch.nn.Linear(400, 300)
        self.l6 = torch.nn.Linear(300, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        x1 = torch.relu(self.l1(xy))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = torch.relu(self.l4(xy))
        x2 = torch.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.max_action = max_action

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=device))

def add_random_noise(action, std):
    return action + np.random.normal(0, std, size=action.shape)

def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1), desc='Collecting demos'):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.select_action(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if (hasattr(env, '_max_episode_steps') and t == env._max_episode_steps) else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
        else:
            state = next_state

    return buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect demonstrations using a trained DAC agent.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where the model is saved')
    parser.add_argument('--model_name', type=str, default='DAC_policy', help='Model name')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Demo buffer size')
    parser.add_argument('--std', type=float, default=0.01, help='Action noise standard deviation')
    parser.add_argument('--p_rand', type=float, default=0.0, help='Probability of random action')
    parser.add_argument('--output_demo_path', type=str, default='demo_collection/DAC_collected_demo.pth', help='Path to save the collected demo')
    args = parser.parse_args()

    env = gym.make('BipedalWalker-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    td3_policy = TD3(state_dim, action_dim, max_action)
    td3_policy.load(args.model_name, args.model_dir)

    demo_buffer = collect_demo(env, td3_policy, buffer_size=args.buffer_size, device=device, std=args.std, p_rand=args.p_rand)
    demo_buffer.save(args.output_demo_path)
    print(f"Saved the new demonstration data to {args.output_demo_path}")
