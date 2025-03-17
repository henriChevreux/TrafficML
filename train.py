import numpy as np
import matplotlib.pyplot as plt
from traffic_env import TrafficGridEnv
from dqn_agent import DQNAgent
import time

def train_dqn_agent(episodes=1000, grid_size=2, batch_size=32, target_update_freq=10):
    """Train the DQN agent on the traffic grid environment."""
    env = TrafficGridEnv(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    agent = DQNAgent(state_size, action_size)
    
    rewards = []
    cars_passed_list = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_cars_passed = 0
        
        for t in range(100):  # Run each episode for 100 steps
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            agent.replay(batch_size)
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            total_cars_passed += reward
            
            if done:
                break
        
        # Periodically update target model
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        rewards.append(total_reward)
        cars_passed_list.append(total_cars_passed)
        
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Cars Passed: {total_cars_passed}, Epsilon: {agent.epsilon:.2f}")
    
    # Save the trained model
    agent.save("traffic_dqn_model.h5")
    
    return rewards, cars_passed_list, agent, env

def evaluate_agent(agent, env, episodes=5):
    """Evaluate the trained agent."""
    total_cars_passed = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_cars = 0
        
        print(f"\nEvaluation Episode {episode + 1}")
        
        for t in range(100):
            # Choose action without exploration
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_cars += reward
            
            # Render environment (print the state)
            if t % 10 == 0:
                env.render()
                print(f"Step {t}, Cars passed in this step: {reward}")
            
            if done:
                break
        
        total_cars_passed.append(episode_cars)
        print(f"Episode {episode + 1} - Total cars passed: {episode_cars}")
    
    return total_cars_passed

def plot_training_results(rewards, cars_passed):
    """Plot the training results."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot total rewards per episode
    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward per Episode')
    
    # Plot cars passed per episode
    ax2.plot(cars_passed)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cars Passed')
    ax2.set_title('Cars Passed per Episode')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define training parameters
    GRID_SIZE = 2  # 2x2 grid of intersections
    EPISODES = 500
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = 10
    
    # Train the agent
    print("Training DQN agent...")
    start_time = time.time()
    rewards, cars_passed, agent, env = train_dqn_agent(
        episodes=EPISODES,
        grid_size=GRID_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training results
    plot_training_results(rewards, cars_passed)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(agent, env)
    print(f"Average cars passed during evaluation: {np.mean(eval_results):.2f}") 