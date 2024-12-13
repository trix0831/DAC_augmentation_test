#!/usr/bin/env python
# coding: utf-8

import gym
import argparse
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.autograd import grad as torch_grad
import h5py
import os
from torch.utils.tensorboard import SummaryWriter
import imageio
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Parse Command Line Arguments
# ----------------------------
parser = argparse.ArgumentParser(description='Train DAC and collect demonstration data.')
parser.add_argument('--num_steps', type=int, default=200000, help='Total training steps')
parser.add_argument('--demo_buffer_size', type=int, default=50000, help='Total steps for the collected demonstration buffer')
args = parser.parse_args()

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
        else:
            pass

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

        self.loss = self.ce_loss

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
        print("Setting adversary learning rate to: {}".format(lr))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def ce_loss(self, pred_on_learner, pred_on_expert, expert_weights):
        learner_loss = torch.log(1 - torch.sigmoid(pred_on_learner))
        expert_loss = torch.log(torch.sigmoid(pred_on_expert)) * expert_weights
        return -torch.sum(learner_loss + expert_loss)

    def learn(self, replay_buf, expert_buf, iterations, batch_size=100):
        self.adjust_adversary_learning_rate(LearningRate.get_instance().lr)
        total_losses = []

        for it in range(iterations):
            x, y, r, d, u = replay_buf.sample(batch_size)
            state = x
            action = y

            expert_obs, expert_act, expert_weights = expert_buf.get_next_batch(batch_size)
            expert_obs = torch.tensor(expert_obs, dtype=torch.float32, device=device)
            expert_act = torch.tensor(expert_act, dtype=torch.float32, device=device)
            expert_weights = torch.tensor(expert_weights, dtype=torch.float32, device=device).view(-1, 1)

            state_action = torch.cat([state, action], 1).to(device)
            expert_state_action = torch.cat([expert_obs, expert_act], 1).to(device)

            min_batch_size = min(state_action.size(0), expert_state_action.size(0))
            state_action = state_action[:min_batch_size]
            expert_state_action = expert_state_action[:min_batch_size]
            expert_weights = expert_weights[:min_batch_size]

            fake = self(state_action)
            real = self(expert_state_action)

            gradient_penalty = self._gradient_penalty(expert_state_action, state_action)
            main_loss = self.loss(fake, real, expert_weights)

            self.optimizer.zero_grad()
            total_loss = main_loss + gradient_penalty
            total_losses.append(total_loss.item())

            if it == 0 or it == iterations - 1:
                print("Discr Iteration:  {:03} ---- Loss: {:.5f} | Learner Prob: {:.5f} | Expert Prob: {:.5f}".format(
                    it, total_loss.item(), torch.sigmoid(fake[0]).item(), torch.sigmoid(real[0]).item()
                ))
            total_loss.backward()
            self.optimizer.step()

        return total_losses

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = min(real_data.size(0), generated_data.size(0))
        device = real_data.device

        alpha = torch.rand(batch_size, 1, device=device)
        interpolated = alpha * real_data[:batch_size] + (1 - alpha) * generated_data[:batch_size]
        interpolated.requires_grad_(True)

        prob_interpolated = self(interpolated)

        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                        grad_outputs=torch.ones_like(prob_interpolated),
                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()

        return self.LAMBDA * gradient_penalty

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        x = x.to(device).float()
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x

    def act(self, x: Tensor) -> Tensor:
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self(x)

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

class TD3(object):
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

    def adjust_critic_learning_rate(self, lr):
        print("Setting critic learning rate to: {}".format(lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def adjust_actor_learning_rate(self, lr):
        print("Setting actor learning rate to: {}".format(lr))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def reward(self, discriminator, states, actions):
        states_actions = torch.cat([states, actions], 1).to(device)
        return discriminator.reward(states_actions)

    def train(self, discriminator, replay_buf, iterations, batch_size=100, discount=0.8, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

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
            reward = self.reward(discriminator, state, action)

            min_batch_size = min(state.size(0), action.size(0), next_state.size(0))
            state = state[:min_batch_size]
            action = action[:min_batch_size]
            next_state = next_state[:min_batch_size]
            reward = reward[:min_batch_size]

            noise = torch.randn_like(action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            if iteration == 0 or iteration == iterations - 1:
                print("Critic Iteration: {:3} ---- Loss: {:.5f}".format(iteration, critic_loss.item()))
            critic_losses.append(critic_loss.item())

            writer.add_scalar('Loss/Critic', critic_loss.item(), lr_tracker.training_step)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if iteration % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                if iteration == 0 or iteration == iterations - 1 or iteration == iterations - 2:
                    print("Actor Iteration:  {:3} ---- Loss: {:.5f}".format(iteration, actor_loss.item()))
                actor_losses.append(actor_loss.item())
                writer.add_scalar('Loss/Actor', actor_loss.item(), lr_tracker.training_step)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                clip_grad_value_(self.actor.parameters(), self.actor_grad_clipping)
                self.actor_optimizer.step()
                lr_tracker.training_step += 1
                step = lr_tracker.training_step

                if step != 0 and step % self.decay_steps == 0:
                    print("Decaying learning rate at step: {}".format(step))
                    lr_tracker.decay()

                    self.adjust_actor_learning_rate(lr_tracker.lr)
                    self.adjust_critic_learning_rate(lr_tracker.lr)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return actor_losses, critic_losses

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


def store_results(evaluations, number_of_timesteps, actor_losses, critic_losses):
    df = pd.DataFrame.from_records(evaluations)
    number_of_trajectories = len(evaluations[0]) - 1
    columns = ["reward_trajectory_{}".format(i + 1) for i in range(number_of_trajectories)]
    columns.append("timestep")
    df.columns = columns

    df['actor_loss'] = pd.Series(actor_losses)
    df['critic_loss'] = pd.Series(critic_losses)

    timestamp = time.time()
    results_fname = 'DAC_{}_tsteps_{}_results.csv'.format(number_of_timesteps, timestamp)
    df.to_csv(str(results_fname), index=False)

def evaluate_policy(env, policy, time_step, evaluation_trajectories=6):
    rewards = []
    for _ in range(evaluation_trajectories):
        r = 0
        obs = env.reset()
        done = False
        while not done:
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            action = policy.select_action(np.array(obs, dtype=np.float32))
            obs, reward, done, info = env.step(action)
            r += reward
        rewards.append(r)
    print("Average reward at timestep {}: {}".format(time_step, np.mean(rewards)))

    rewards.append(time_step)
    return rewards

def load_dataset(path, limit_trajs=None, data_subsamp_freq=1):
    tmp = torch.load(path)
    full_dset_size = tmp['state'].size(0)
    state_dim = tmp['state'].size(1)
    action_dim = tmp['action'].size(1)

    steps_per_traj = 500
    num_trajs = full_dset_size // steps_per_traj
    dset_size = num_trajs * steps_per_traj

    states = tmp['state'][:dset_size].reshape(num_trajs, steps_per_traj, state_dim).clone()
    actions = tmp['action'][:dset_size].reshape(num_trajs, steps_per_traj, action_dim).clone()
    rewards = tmp['reward'][:dset_size].reshape(num_trajs, steps_per_traj, 1).clone()
    dones = tmp['done'][:dset_size].reshape(num_trajs, steps_per_traj, 1).clone()
    next_states = tmp['next_state'][:dset_size].reshape(num_trajs, steps_per_traj, state_dim).clone()
    return states, actions, rewards

class Dset(object):
    def __init__(self, obs, acs, num_traj, absorbing_state, absorbing_action):
        self.obs = obs
        self.acs = acs
        self.num_traj = num_traj
        assert len(self.obs) == len(self.acs)
        assert self.num_traj > 0
        self.steps_per_traj = int(len(self.obs) / num_traj)

        self.absorbing_state = absorbing_state
        self.absorbing_action = absorbing_action

    def get_next_batch(self, batch_size):
        assert batch_size <= len(self.obs)
        num_samples_per_traj = max(1, batch_size // self.num_traj)

        if num_samples_per_traj * self.num_traj != batch_size:
            batch_size = num_samples_per_traj * self.num_traj

        N = self.steps_per_traj / num_samples_per_traj
        j = num_samples_per_traj
        num_samples_per_traj = num_samples_per_traj - 1

        obs = None
        acs = None
        weights = [1 for i in range(batch_size)]
        while j <= batch_size:
            weights[j - 1] = 1.0 / N
            j = j + num_samples_per_traj + 1

        for i in range(self.num_traj):
            indicies = np.sort(
                np.random.choice(range(self.steps_per_traj * i, self.steps_per_traj * (i + 1)), num_samples_per_traj,
                                 replace=False))
            if obs is None:
                obs = np.concatenate((self.obs[indicies, :], self.absorbing_state), axis=0)
            else:
                obs = np.concatenate((obs, self.obs[indicies, :], self.absorbing_state), axis=0)

            if acs is None:
                acs = np.concatenate((self.acs[indicies, :], self.absorbing_action), axis=0)
            else:
                acs = np.concatenate((acs, self.acs[indicies, :], self.absorbing_action), axis=0)

        return obs, acs, weights

class Mujoco_Dset(object):
    def __init__(self, env, expert_path, traj_limitation=-1):
        obs, acs, rets = load_dataset(expert_path, traj_limitation)
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = rets.sum(axis=1)
        try:
            self.avg_ret = sum(self.rets) / len(self.rets)
        except:
            self.avg_ret = 0
        self.std_ret = np.std(np.array(self.rets))
        assert len(self.obs) == len(self.acs)
        self.num_traj = len(rets)
        self.num_transition = len(self.obs)

        absorbing_state = np.zeros((1,env.observation_space.shape[0]), dtype=np.float32)
        zero_action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
        self.dset = Dset(self.obs, self.acs, self.num_traj, absorbing_state, zero_action)
        self.log_info()

    def log_info(self):
        print("Total trajs: %d" % self.num_traj)
        print("Total transitions: %d" % self.num_transition)
        print("Average returns: %f" % self.avg_ret)
        print("Std for returns: %f" % self.std_ret)

    def get_next_batch(self, batch_size):
        return self.dset.get_next_batch(batch_size)

# Collect demo function as per user request:
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

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)  # deterministic action
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

def add_random_noise(action, std):
    return action + np.random.normal(0, std, size=action.shape)

# --------------------------
# Main Training Script
# --------------------------
env = gym.make('BipedalWalker-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

trajectory_length = 2500
batch_size = 2500
num_steps = args.num_steps  # controlled by --num_steps
demo_buffer_size = args.demo_buffer_size  # controlled by --demo_buffer_size

lr = LearningRate.get_instance()
lr.lr = 10 ** (-3)
lr.decay_factor = 0.5

expert_buffer = Mujoco_Dset(env, 'demo_collection/Bipedal_size5000_std0.01_prand0.0.pth', 5000)
state_shape = env.observation_space.shape
action_shape = env.action_space.shape

actor_replay_buffer = Buffer(buffer_size=num_steps, state_shape=state_shape, action_shape=action_shape, device=device)

td3_policy = TD3(state_dim, action_dim, max_action, 40, 10 ** 5)
discriminator = Discriminator(state_dim + action_dim).to(device)

# Start timing
start_time = time.time()

writer = SummaryWriter()
evaluations = [evaluate_policy(env, td3_policy, 0)]
evaluate_every = 8000
steps_since_eval = 0

env.reset()

actor_losses = []
critic_losses = []

while len(actor_replay_buffer) < num_steps:
    print("\nCurrent step: {}".format(len(actor_replay_buffer)))
    current_state = env.reset()
    for j in range(trajectory_length):
        action = td3_policy.select_action(np.array(current_state))
        obs, reward, done, info = env.step(action)

        if done:
            actor_replay_buffer.addAbsorbing()
            current_state = env.reset()
        else:
            actor_replay_buffer.add((current_state, action, obs), reward, done)
            current_state = obs

    discriminator_losses = discriminator.learn(actor_replay_buffer, expert_buffer, trajectory_length, batch_size)
    td3_actor_losses, td3_critic_losses = td3_policy.train(discriminator, actor_replay_buffer, trajectory_length, batch_size)

    actor_losses.extend(td3_actor_losses)
    critic_losses.extend(td3_critic_losses)

    if steps_since_eval >= evaluate_every:
        steps_since_eval = 0
        evaluation = evaluate_policy(env, td3_policy, len(actor_replay_buffer))
        evaluations.append(evaluation)

    steps_since_eval += trajectory_length

last_evaluation = evaluate_policy(env, td3_policy, len(actor_replay_buffer))
evaluations.append(last_evaluation)

store_results(evaluations, len(actor_replay_buffer), actor_losses, critic_losses)
writer.close()

# Record a 1500 steps video of the trained agent
frames = []
num_episodes_record = 5
max_steps_per_episode = 1500
video_env = gym.make('BipedalWalker-v3')

for episode in range(num_episodes_record):
    state = video_env.reset()
    done = False
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        action = td3_policy.select_action(np.array(state))
        state, reward, done, info = video_env.step(action)
        
        frame = video_env.render(mode='rgb_array')
        frames.append(frame)
        
        steps += 1

video_env.close()

video_filename = 'trained_agent_video_5_episodes.mp4'
imageio.mimsave(video_filename, frames, fps=30)

# After training, collect demonstration data for demo_buffer_size steps
# For simplicity, define exploit method as same as select_action
td3_policy.exploit = lambda s: td3_policy.select_action(np.array(s))
demo_std = 0.01   # Some standard deviation for action noise
p_rand = 0.0      # Probability of random action, adjust as needed

demo_buffer = collect_demo(env, td3_policy, buffer_size=demo_buffer_size, device=device, std=demo_std, p_rand=p_rand)
output_demo_path = f"demo_collection/DAC_aug_buffer_size_{demo_buffer_size}_from_5000.pth"
demo_buffer.save(output_demo_path)
print(f"Saved the new demonstration-like data to {output_demo_path}")

# End timing
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime:.2f} seconds")
writer.add_text('Runtime', f"{total_runtime:.2f} seconds")
