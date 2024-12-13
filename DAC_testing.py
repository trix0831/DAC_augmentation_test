#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
import torch
import gym
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(device)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

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

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth"), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth"), map_location=device))

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a trained DAC agent with multiple seeds.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where the model is saved')
    parser.add_argument('--model_name', type=str, default='DAC_policy', help='Model name prefix')
    parser.add_argument('--num_tests', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--max_length', type=int, default=1500, help='Maximum steps per episode during testing')
    parser.add_argument('--seeds', type=str, default='0', help='Comma-separated list of random seeds for testing')
    parser.add_argument('--output_csv', type=str, default='test_results.csv', help='Path to the output CSV file')
    parser.add_argument('--hardcore', type=bool, default=False, help='Use the BipedalWalkerHardcore-v3 environment if set')
    args = parser.parse_args()

    # Parse seeds
    seeds = list(map(int, args.seeds.split(',')))
    if len(seeds) != args.num_tests:
        raise ValueError("Number of seeds must match the number of tests.")

    # Select environment
    env_name = 'BipedalWalkerHardcore-v3' if args.hardcore else 'BipedalWalker-v3'
    env = gym.make(env_name)
    # , hardcore=args.hardcore

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Load the trained policy
    policy = TD3(state_dim, action_dim, max_action)
    policy.load(args.model_name, args.model_dir)

    # Test the agent
    results = []
    for i in range(args.num_tests):
        seed = seeds[i]

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        env.seed(seed)

        state = env.reset()
        done = False
        episode_return = 0.0
        step_count = 0

        for t in range(args.max_length):
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            episode_return += reward
            step_count += 1
            if done:
                break

        results.append({
            'episode': i + 1,
            'seed': seed,
            'return': episode_return,
            'steps': step_count
        })

    env.close()

    # Save results to CSV
    df = pd.DataFrame(results, columns=['episode', 'seed', 'return', 'steps'])
    df.to_csv(args.output_csv, index=False)
    print(f"Test results saved to {args.output_csv}")
