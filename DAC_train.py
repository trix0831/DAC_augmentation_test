#!/usr/bin/env python
# coding: utf-8

import gym
import argparse
import numpy as np
import pandas as pd
import torch
import time
import os
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearningRate:
    """
    Singleton for maintaining learning rate and step count.
    """
    __instance = None

    def __init__(self):
        if LearningRate.__instance is not None:
            raise Exception("Singleton instantiation called twice")
        else:
            LearningRate.__instance = self
            self.lr = None
            self.decay_factor = None
            self.training_step = 0

    @staticmethod
    def get_instance():
        if LearningRate.__instance is None:
            LearningRate()
        return LearningRate.__instance

    def set_learning_rate(self, lr):
        self.lr = lr

    def get_learning_rate(self):
        return self.lr

    def increment_step(self):
        self.training_step += 1

    def get_step(self):
        return self.training_step

    def set_decay(self, d):
        self.decay_factor = d

    def decay(self):
        if self.lr is None:
            raise ValueError("Learning rate has not been set.")
        self.lr = self.lr * self.decay_factor


class SerializedBuffer:
    def __init__(self, path=None, device=None):
        if path is not None:
            tmp = torch.load(path)
            self.buffer_size = self._n = tmp['state'].size(0)
            self.device = device

            self.states = tmp['state'].clone().to(self.device)
            self.actions = tmp['action'].clone().to(self.device)
            self.rewards = tmp['reward'].clone().to(self.device)
            self.dones = tmp['done'].clone().to(self.device)
            self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):
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

    def __len__(self):
        return self._n

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

    def save_contents(self):
        return {
            'state': self.states[:self._n].cpu(),
            'action': self.actions[:self._n].cpu(),
            'reward': self.rewards[:self._n].cpu(),
            'done': self.dones[:self._n].cpu(),
            'next_state': self.next_states[:self._n].cpu(),
            '_n': self._n,
            '_p': self._p
        }

    def load_contents(self, data):
        length = data['_n']
        self._n = length
        self._p = data['_p']
        self.states[:length] = data['state'].to(self.device)
        self.actions[:length] = data['action'].to(self.device)
        self.rewards[:length] = data['reward'].to(self.device)
        self.dones[:length] = data['done'].to(self.device)
        self.next_states[:length] = data['next_state'].to(self.device)


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=100, lamb=10, entropy_weight=0.001):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
        self.criterion = nn.BCEWithLogitsLoss()
        self.entropy_weight = entropy_weight
        self.optimizer = torch.optim.Adam(self.parameters())
        self.LAMBDA = lamb
        self.use_cuda = torch.cuda.is_available()

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        out = self.linear3(x)
        return out

    def reward(self, x):
        out = self(x)
        probs = torch.sigmoid(out)
        return torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)

    def adjust_adversary_learning_rate(self, lr):
        # print("Setting adversary learning rate to: {}".format(lr))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device).float()
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

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        xy = torch.cat([x, y], dim=1)

        x1 = torch.relu(self.l1(xy))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = torch.relu(self.l4(xy))
        x2 = torch.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, x: Tensor, y: Tensor) -> Tensor:
        xy = torch.cat([x, y], dim=1)
        x1 = torch.relu(self.l1(xy))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1

class TD3:
    def __init__(self, state_dim, action_dim, max_action, actor_clipping, decay_steps):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.decay_steps = decay_steps
        self.actor_grad_clipping = actor_clipping
        self.max_action = max_action

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, discriminator, replay_buf, iterations, batch_size=100, discount=0.8, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2, writer=None):

        lr_tracker = LearningRate.get_instance()
        lr = lr_tracker.lr

        self.adjust_actor_learning_rate(lr)
        self.adjust_critic_learning_rate(lr)

        actor_losses = []
        critic_losses = []

        for iteration in range(iterations):
            x, y, r, d, u = replay_buf.sample(batch_size)
            state = x
            action = y
            next_state = u
            # Compute reward using discriminator
            reward = discriminator.reward(torch.cat([state, action], dim=1).to(device))

            # Clipped next action with noise
            next_action = self.actor_target(next_state) + \
                torch.clamp(torch.randn_like(action) * policy_noise, -noise_clip, noise_clip)
            next_action = torch.clamp(next_action, -self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + discount * torch.min(target_Q1, target_Q2).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_losses.append(critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Policy update
            if iteration % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                actor_losses.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_value_(self.actor.parameters(), self.actor_grad_clipping)
                self.actor_optimizer.step()

                # Update the target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return actor_losses, critic_losses

    def adjust_actor_learning_rate(self, lr):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def adjust_critic_learning_rate(self, lr):
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def save(self, filename, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth"), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth"), map_location=device))


def store_results(evaluations, number_of_timesteps, actor_losses, critic_losses, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame.from_records(evaluations)
    columns = [f"reward_trajectory_{i+1}" for i in range(len(evaluations[0]) - 1)] + ["timestep"]
    df.columns = columns
    df["actor_loss"] = pd.Series(actor_losses)
    df["critic_loss"] = pd.Series(critic_losses)

    timestamp = int(time.time())
    results_file = os.path.join(results_dir, f"DAC_results_{number_of_timesteps}_steps_{timestamp}.csv")
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


def evaluate_policy(env, policy, time_step, evaluation_trajectories=6):
    rewards = []
    for _ in range(evaluation_trajectories):
        r = 0
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, done, _ = env.step(action)
            r += reward
        rewards.append(r)
    rewards.append(time_step)
    return rewards


def save_checkpoint(checkpoint_path, td3_policy, discriminator, replay_buffer, actor_losses, critic_losses, evaluations, steps_since_eval, lr_tracker):
    checkpoint = {
        'actor_state_dict': td3_policy.actor.state_dict(),
        'critic_state_dict': td3_policy.critic.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'replay_buffer': replay_buffer.save_contents(),
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'evaluations': evaluations,
        'steps_since_eval': steps_since_eval,
        'lr': lr_tracker.lr,
        'decay_factor': lr_tracker.decay_factor,
        'training_step': lr_tracker.training_step
    }

    directory = os.path.dirname(checkpoint_path)
    if directory and directory.strip():
        os.makedirs(directory, exist_ok=True)
    # If directory is empty, it means checkpoint_path has no directory component and is just a filename in current dir

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, td3_policy, discriminator, replay_buffer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    td3_policy.actor.load_state_dict(checkpoint['actor_state_dict'])
    td3_policy.critic.load_state_dict(checkpoint['critic_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    replay_buffer.load_contents(checkpoint['replay_buffer'])

    actor_losses = checkpoint['actor_losses']
    critic_losses = checkpoint['critic_losses']
    evaluations = checkpoint['evaluations']
    steps_since_eval = checkpoint['steps_since_eval']

    lr_tracker = LearningRate.get_instance()
    lr_tracker.lr = checkpoint['lr']
    lr_tracker.decay_factor = checkpoint['decay_factor']
    lr_tracker.training_step = checkpoint['training_step']

    print(f"Loaded checkpoint from {checkpoint_path}")
    return actor_losses, critic_losses, evaluations, steps_since_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAC agent.")
    parser.add_argument("--num_steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--expert_buffer_path", type=str, required=True, help="Path to expert buffer")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint", help="Path to save/load checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=50000, help="Interval to save checkpoints")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to resume training from a checkpoint")

    args = parser.parse_args()
    
    print(f"checkpoint_path: {args.checkpoint_path}")

    num_steps = args.num_steps
    results_dir = args.results_dir
    model_dir = args.model_dir
    expert_buffer_path = args.expert_buffer_path
    checkpoint_path = args.checkpoint_path
    checkpoint_interval = args.checkpoint_interval
    resume_path = args.resume_path

    env = gym.make("BipedalWalker-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    trajectory_length = 2500
    batch_size = 2500

    lr = LearningRate.get_instance()
    lr.set_learning_rate(1e-3)
    lr.set_decay(0.5)

    expert_buffer = SerializedBuffer(expert_buffer_path, device)

    replay_buffer = Buffer(num_steps, env.observation_space.shape, env.action_space.shape, device)
    policy = TD3(state_dim, action_dim, max_action, actor_clipping=40, decay_steps=100000)
    discriminator = Discriminator(state_dim + action_dim).to(device)

    writer = SummaryWriter()
    evaluations = [evaluate_policy(env, policy, 0)]
    steps_since_eval = 0

    actor_losses = []
    critic_losses = []

    # Resume if required
    if resume_path is not None and resume_path.strip():
        a_losses, c_losses, evaluations, steps_since_eval = load_checkpoint(resume_path, policy, discriminator, replay_buffer)
        actor_losses = a_losses
        critic_losses = c_losses

    while len(replay_buffer) < num_steps:
        print(f"Current buffer size: {len(replay_buffer)}, Last evaluation mean reward: {np.mean(evaluations[-1][:-1]) if len(evaluations[-1])>1 else 0}")   
        
        current_state = env.reset()
        for _ in range(trajectory_length):
            action = policy.select_action(current_state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append(current_state, action, reward, done, next_state)

            if done:
                current_state = env.reset()
            else:
                current_state = next_state

        # Adjust discriminator learning rate
        lr_tracker = LearningRate.get_instance()
        discriminator.adjust_adversary_learning_rate(lr_tracker.lr)

        # Train the discriminator for 'trajectory_length' iterations as a placeholder
        for _ in range(trajectory_length):
            x, y, r, d, u = replay_buffer.sample(batch_size)
            state = x
            action = y

            # sample expert data
            idxes = np.random.randint(low=0, high=expert_buffer._n, size=batch_size)
            expert_obs = expert_buffer.states[idxes]
            expert_act = expert_buffer.actions[idxes]
            # Assign equal weights
            expert_weights = torch.ones((batch_size,1), device=device, dtype=torch.float32)

            state_action = torch.cat([state, action], 1).to(device)
            expert_state_action = torch.cat([expert_obs, expert_act], 1).to(device)

            min_batch_size = min(state_action.size(0), expert_state_action.size(0))
            state_action = state_action[:min_batch_size]
            expert_state_action = expert_state_action[:min_batch_size]
            expert_weights = expert_weights[:min_batch_size]

            fake = discriminator(state_action)
            real = discriminator(expert_state_action)

            # gradient penalty
            batch_size_ = min_batch_size
            alpha = torch.rand(batch_size_, 1, device=device)

            interpolated = alpha * expert_state_action[:batch_size_] + (1 - alpha) * state_action[:batch_size_]
            interpolated.requires_grad_(True)

            prob_interpolated = discriminator(interpolated)
            gradients = torch.autograd.grad(
                outputs=prob_interpolated, 
                inputs=interpolated,
                grad_outputs=torch.ones_like(prob_interpolated),
                create_graph=True, 
                retain_graph=True
            )[0]

            gradients = gradients.view(batch_size_, -1)
            gradients_norm = gradients.norm(2, dim=1)
            gradient_penalty = ((gradients_norm - 1) ** 2).mean() * discriminator.LAMBDA

            # CE loss
            learner_loss = torch.log(1 - torch.sigmoid(fake))
            expert_loss = torch.log(torch.sigmoid(real)) * expert_weights
            main_loss = -torch.sum(learner_loss + expert_loss)

            total_loss = main_loss + gradient_penalty
            discriminator.optimizer.zero_grad()
            total_loss.backward()
            discriminator.optimizer.step()

        # Train the TD3 policy
        td3_actor_losses, td3_critic_losses = policy.train(discriminator, replay_buffer, trajectory_length, batch_size, writer=writer)
        actor_losses.extend(td3_actor_losses)
        critic_losses.extend(td3_critic_losses)

        steps_since_eval += trajectory_length
        if steps_since_eval >= 8000:
            evaluations.append(evaluate_policy(env, policy, len(replay_buffer)))
            steps_since_eval = 0

        if checkpoint_path and checkpoint_path.strip():
            if len(replay_buffer) % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, policy, discriminator, replay_buffer, actor_losses, critic_losses, evaluations, steps_since_eval, LearningRate.get_instance())
                print(f"Checkpoint saved at {len(replay_buffer)} steps")

    evaluations.append(evaluate_policy(env, policy, len(replay_buffer)))
    store_results(evaluations, len(replay_buffer), actor_losses, critic_losses, results_dir)

    model_name = f"DAC_policy_{num_steps}_steps_{int(time.time())}"
    policy.save(model_name, model_dir)
    print(f"Model saved as {model_name} in {model_dir}")

    # Final checkpoint
    if checkpoint_path and checkpoint_path.strip():
        save_checkpoint(checkpoint_path, policy, discriminator, replay_buffer, actor_losses, critic_losses, evaluations, steps_since_eval, LearningRate.get_instance())
        print(f"Final checkpoint saved at {len(replay_buffer)} steps")
