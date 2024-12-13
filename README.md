# README

This repository provides three Python scripts for training a DAC (Discriminator Actor-Critic) agent, collecting demonstration data from a trained agent, and recording a video of the trained agent interacting with an environment. The environment used is typically `BipedalWalker-v3` from OpenAI Gym, but you can adapt the code to other environments that have compatible observation and action spaces.

## Overview

1. **train.py**:  
   Trains a DAC agent (TD3 + Discriminator) using expert data. After training, the model parameters (actor and critic) are saved to the specified directory.
   
2. **collect_demo.py**:  
   Loads a trained DAC agent model and uses it to collect demonstration data (state, action, reward, etc.) for a desired number of steps. The collected data is stored for future use.
   
3. **record_video.py**:  
   Loads a trained DAC agent model and records a video of the agent acting in the environment. The recorded frames are saved as a video file.

## Dependencies

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [gym](https://github.com/openai/gym)
- [imageio](https://imageio.github.io/)
- [tqdm](https://github.com/tqdm/tqdm)
- [pandas](https://pandas.pydata.org/)
- [tensorboard](https://www.tensorflow.org/tensorboard)

Make sure your environment is properly set up with these libraries before running the scripts.

## Usage

### 1. Training the DAC Agent

**Script:** `train.py`

**Function:** Trains a DAC agent using expert demonstration data.

**Arguments:**

- `--num_steps`: The total number of training steps to run. Default is `200000`.

**Example Command:**
```bash
python train.py --num_steps 5000
```

After training, the saved model weights (`DAC_policy_actor.pth` and `DAC_policy_critic.pth`) will be placed in the `models` directory by default.

### 2. Collecting Demonstration Data

**Script:** `collect_demo.py`

**Function:** Uses the trained DAC model to collect demonstration data. The agent will run in the environment for a specified number of steps, and the trajectory data (states, actions, rewards, etc.) will be saved to a file for later use.

**Arguments:**

- `--model_dir`: Directory where the trained model files are stored. Default: `models`.
- `--model_name`: Prefix name of the saved model files. Default: `DAC_policy`.
- `--buffer_size`: Number of steps of demonstration data to collect. Default: `50000`.
- `--std`: Standard deviation of action noise added to the policy’s actions. Default: `0.01`.
- `--p_rand`: Probability of taking a random action instead of the policy’s action. Default: `0.0`.
- `--output_demo_path`: Path to save the collected demonstration data. Default: `demo_collection/DAC_collected_demo.pth`.

**Example Command:**
```bash
python collect_demo.py --model_dir models --model_name DAC_policy --buffer_size 50000 --std 0.01 --p_rand 0.0 --output_demo_path demo_collection/DAC_collected_demo.pth
```

This will produce a `.pth` file with the collected demonstration data.

### 3. Recording a Video

**Script:** `record_video.py`

**Function:** Renders and records the trained agent’s behavior in the environment to a video file.

**Arguments:**

- `--model_dir`: Directory where the trained model files are stored. Default: `models`.
- `--model_name`: Prefix name of the saved model files. Default: `DAC_policy`.
- `--episodes`: Number of episodes to record. Default: `5`.
- `--max_steps_per_episode`: Maximum number of steps per episode to record. Default: `1500`.
- `--output_video_path`: File path where the recorded video will be saved. Default: `trained_agent_video.mp4`.

**Example Command:**
```bash
python record_video.py --model_dir models --model_name DAC_policy --episodes 5 --max_steps_per_episode 1500 --output_video_path trained_agent_video.mp4
```

This will generate a video file showing the agent’s performance.

## Suggested Workflow

1. **Train the agent** using `train.py` with your desired `--num_steps`.  
2. **Collect demonstrations** using `collect_demo.py` from the newly trained model.  
3. **Record a video** of the trained agent’s behavior using `record_video.py` to visualize performance.

By following the above steps, you can iterate on your training process, gather demonstration data, and visually inspect how the trained agent performs.