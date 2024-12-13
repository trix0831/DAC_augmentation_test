# README

This repository contains several Python scripts that together demonstrate the following workflow for a DAC (Discriminator Actor-Critic) agent:

1. **Training the DAC Agent** (`train.py`):  
   Trains a DAC (TD3 + Discriminator) agent using expert demonstration data. Produces model weights that can be loaded by other scripts.

2. **Collecting Demonstration Data** (`collect_demo.py`):  
   Uses the trained DAC agent to interact with the environment and produce new demonstration data for future training or analysis.

3. **Recording a Video** (`record_video.py`):  
   Loads a trained DAC agent and records its behavior in a video file. Now supports a `--hardcore` flag to run the `BipedalWalkerHardcore-v3` environment.

4. **Testing the Agent** (`testing.py`):  
   Loads a trained DAC agent and evaluates its performance across multiple test episodes. Users can specify a list of seeds, maximum steps per episode, and number of tests. Saves test results to a CSV file. Also supports a `--hardcore` flag to test in the hardcore environment.

## Dependencies

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [gym](https://github.com/openai/gym)
- [imageio](https://imageio.github.io/)
- [pandas](https://pandas.pydata.org/)
- [tensorboard](https://www.tensorflow.org/tensorboard)
- [tqdm](https://github.com/tqdm/tqdm)

## Usage

### 1. Train the Agent: `train.py`

**Arguments:**
- `--num_steps`: Total number of training steps (default: `200000`)

**Example:**
```bash
python train.py --num_steps 5000
```

Output: Saves model weights in `models/` directory by default.

### 2. Collect Demonstration Data: `collect_demo.py`

**Arguments:**
- `--model_dir`: Directory of saved model files (default: `models`)
- `--model_name`: Model name prefix (default: `DAC_policy`)
- `--buffer_size`: Steps of data to collect (default: `50000`)
- `--std`: Action noise standard deviation (default: `0.01`)
- `--p_rand`: Probability of random action (default: `0.0`)
- `--output_demo_path`: Path for the collected demonstration data file

**Example:**
```bash
python collect_demo.py --model_dir models --model_name DAC_policy --buffer_size 50000 --std 0.01 --p_rand 0.0 --output_demo_path demo_collection/my_demo.pth
```

Output: Saves a `.pth` file with collected demonstration data.

### 3. Record a Video: `record_video.py`

**Arguments:**
- `--model_dir`: Directory of saved model files (default: `models`)
- `--model_name`: Model name prefix (default: `DAC_policy`)
- `--episodes`: Number of episodes to record (default: `5`)
- `--max_steps_per_episode`: Max steps per episode (default: `1500`)
- `--output_video_path`: Output video file path (default: `trained_agent_video.mp4`)
- `--hardcore`: Use `BipedalWalkerHardcore-v3` environment if set.

**Example:**
```bash
python record_video.py --model_dir models --model_name DAC_policy --episodes 2 --max_steps_per_episode 1000 --output_video_path agent_video.mp4 --hardcore
```

Output: A `agent_video.mp4` file showing the agent in the hardcore environment.

### 4. Test the Agent: `testing.py`

**Arguments:**
- `--model_dir`: Directory of saved model files (default: `models`)
- `--model_name`: Model name prefix (default: `DAC_policy`)
- `--num_tests`: Number of test episodes (default: `10`)
- `--max_length`: Max steps per test episode (default: `1500`)
- `--seeds`: Comma-separated list of seeds. The number of seeds must match `--num_tests`.
- `--output_csv`: CSV file path for results (default: `test_results.csv`)
- `--hardcore`: Use `BipedalWalkerHardcore-v3` environment if set.

**Example:**
```bash
python testing.py --model_dir models --model_name DAC_policy --num_tests 3 --max_length 1000 --seeds 0,10,42 --hardcore --output_csv test_results.csv
```

Output: A `test_results.csv` file containing the returns and steps taken in each test episode using the specified seeds.

---

## Suggested Workflow

1. **Train the agent** using `train.py`.
2. **Collect additional demonstrations** using `collect_demo.py`.
3. **Record a video** of the trained agent using `record_video.py` for qualitative evaluation.
4. **Test the agent** with various seeds and optional hardcore environment using `testing.py` and analyze the results in the generated CSV file.

This process allows you to develop, evaluate, and visualize the performance of the DAC agent.