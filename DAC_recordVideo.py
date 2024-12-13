#!/usr/bin/env python
# coding: utf-8

import gym
import argparse
import numpy as np
import torch
import imageio
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record a video of the trained agent.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where the model is saved')
    parser.add_argument('--model_name', type=str, default='DAC_policy', help='Model name')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to record')
    parser.add_argument('--max_steps_per_episode', type=int, default=1500, help='Max steps per episode')
    parser.add_argument('--output_video_dir', type=str, default='video', help='Output video directory')
    parser.add_argument('--hardcore', type=bool, default=False, help='Use BipedalWalkerHardcore-v3 environment if set')
    args = parser.parse_args()

    os.makedirs(args.output_video_dir, exist_ok=True)

    env_name = 'BipedalWalkerHardcore-v3' if args.hardcore else 'BipedalWalker-v3'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    td3_policy = TD3(state_dim, action_dim, max_action)
    td3_policy.load(args.model_name, args.model_dir)

    for episode in range(args.episodes):
        frames = []
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < args.max_steps_per_episode:
            action = td3_policy.select_action(np.array(state))
            state, reward, done, info = env.step(action)
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            steps += 1

        video_filename = os.path.join(
            args.output_video_dir, 
            f'video/trained_agent_episode_{episode + 1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        )
        imageio.mimsave(video_filename, frames, fps=30)
        print(f"Episode {episode + 1} video saved to {video_filename}")

    env.close()
