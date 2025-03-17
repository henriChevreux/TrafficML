# TrafficML: Traffic Light Optimization with DQN

This project implements a Deep Q-Network (DQN) reinforcement learning model to optimize traffic lights in a grid pattern. The goal is to maximize the number of cars passing through intersections.

## Project Overview

The system models a grid of intersections, each controlled by a traffic light that can be in one of two phases:
- North-South Green (East-West Red)
- East-West Green (North-South Red)

The DQN agent learns a policy to control these traffic lights to minimize queue lengths and maximize throughput.

## Components

1. **Traffic Environment** (`traffic_env.py`)
   - A custom OpenAI Gym environment that simulates a grid of traffic intersections
   - Models vehicle arrivals, queue formation, and traffic flow based on light phases
   - Provides observations of queue lengths and traffic light states
   - Rewards the agent based on the number of vehicles passing through intersections

2. **DQN Agent** (`dqn_agent.py`)
   - Implements the Deep Q-Network algorithm
   - Uses experience replay to learn from past experiences
   - Employs a target network to stabilize training
   - Includes epsilon-greedy exploration strategy

3. **Training Script** (`train.py`)
   - Trains the DQN agent on the traffic environment
   - Evaluates the trained agent's performance
   - Plots training metrics (rewards and cars passed)

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/TrafficML.git
   cd TrafficML
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the DQN agent:
   ```
   python train.py
   ```

2. The training script will:
   - Train the agent for 500 episodes
   - Save the trained model weights to `traffic_dqn_model.h5`
   - Plot training metrics and save them to `training_results.png`
   - Evaluate the trained agent's performance

## Customization

You can customize various parameters in `train.py`:
- `GRID_SIZE`: Size of the traffic grid (default: 2x2)
- `EPISODES`: Number of training episodes (default: 500)
- `BATCH_SIZE`: Batch size for experience replay (default: 32)
- `TARGET_UPDATE_FREQ`: Frequency of target network updates (default: 10 episodes)

## Future Improvements

- Add more realistic traffic modeling
- Implement more complex traffic light phases
- Explore different RL algorithms (e.g., DDPG, PPO)
- Add visualization of the traffic grid
- Incorporate real-world traffic data
