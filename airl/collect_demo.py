import os
import argparse
import torch
import gym

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.utils import collect_demo
from gym.wrappers import Monitor

def run(args):
    # First, collect demonstrations
    env = make_env(args.env_id)
    device = torch.device("cuda" if args.cuda else "cpu")

    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        path=args.weight
    )

    buffer = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    )
    save_path = os.path.join(
        'buffers',
        args.env_id,
        f'size{args.buffer_size}_std{args.std}_prand{args.p_rand}.pth'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    buffer.save(save_path)

    # Close the environment used for data collection
    env.close()

    # Now, record 3 evaluation episodes
    video_dir = os.path.join('videos', args.env_id)
    os.makedirs(video_dir, exist_ok=True)

    # Re-initialize environment for recording (Gym 0.21.0)
    env = make_env(args.env_id)
    # Wrap with Monitor for video recording
    env = Monitor(
        env, 
        directory=video_dir, 
        video_callable=lambda episode_id: True, 
        force=True
    )

    # Run the policy for 3 episodes and record them
    for episode in range(0):
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                # Just call exploit and use the returned value directly
                action = algo.exploit(obs)
            
            # If action is a scalar float, and you need it as array, do:
            # action = np.array([action])  # only if necessary

            obs, reward, done, info = env.step(action)



    env.close()

    print(f"Demonstrations saved to {save_path}")
    print(f"Videos recorded in {video_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
