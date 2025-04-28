import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

# Create Taxi environment
env = gym.make("Taxi-v3").env

# Q-table represents the rewards (Q-values) the agent can expect performing a certain action in a certain state
state_space = env.observation_space.n  # total number of states
action_space = env.action_space.n  # total number of actions
qtable = np.zeros((state_space, action_space))  # initialize Q-table with zeros

# Variables for training/testing
test_episodes = 20000  # number of episodes for testing
train_episodes = 60000  # number of episodes for training
episodes = train_episodes + test_episodes  # total number of episodes
max_steps = 100  # maximum number of steps per episode

# Q-learning algorithm hyperparameters to tune
alpha = 0.35  # learning rate: you may change it to see the difference
gamma = 0.75  # discount factor: you may change it to see the difference

# Exploration-exploitation trade-off
epsilon = 1.0  # probability the agent will explore (initial value is 1.0)
epsilon_min = 0.001  # minimum value of epsilon 
epsilon_decay = 0.9999  # decay multiplied with epsilon after each episode

# Lists to hold rewards and steps per episode
training_rewards = []
training_steps = []
testing_rewards = []
testing_steps = []

# Training phase
for episode in range(train_episodes):
    state = env.reset()[0]  
    total_reward = 0
    steps = 0
    
    for _ in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state]) 
            
        next_state, reward, terminated, truncated, _ = env.step(action)  # take the action

        done = terminated or truncated
        
        # Update Q-table using the Q-learning formula
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[next_state]) - qtable[state, action])
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
            
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    # Record training rewards and steps
    training_rewards.append(total_reward)
    training_steps.append(steps)

print("Training done.")

# Testing phase
for episode in range(test_episodes):
    state = env.reset()[0]  # reset state 
    total_reward = 0
    steps = 0
    
    for _ in range(max_steps):
        action = np.argmax(qtable[state])
        next_state, reward, terminated, truncated, _ = env.step(action)  

        done = terminated or truncated
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
            
    # Record testing rewards and steps
    testing_rewards.append(total_reward)
    testing_steps.append(steps)

print("Testing finished.\n")

# Plot rewards and steps
def plot_metrics(training_rewards, training_steps, testing_rewards, testing_steps):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(training_rewards)
    axs[0, 0].set_title('Training Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    axs[0, 1].plot(training_steps)
    axs[0, 1].set_title('Training Steps')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    
    axs[1, 0].plot(testing_rewards)
    axs[1, 0].set_title('Testing Rewards')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Reward')
    
    axs[1, 1].plot(testing_steps)
    axs[1, 1].set_title('Testing Steps')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

plot_metrics(training_rewards, training_steps, testing_rewards, testing_steps)
