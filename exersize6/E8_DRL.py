# Import libraries
import gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DQNAgent():
    def __init__(self, env_id, epsilon_decay=0.9999, epsilon=1.0, epsilon_min=0.001, 
            gamma=0.95, alpha=0.01, alpha_decay=0.01, batch_size=16):

        self.memory = deque(maxlen=50000) # memory length
        self.env = gym.make(env_id) # create environment with env_id

        self.state_size = self.env.observation_space.shape[0] 
        self.action_size = self.env.action_space.n # total number of actions

        self.epsilon = epsilon # probability the agent will explore (initial value is 1.0)
        self.epsilon_decay = epsilon_decay # decay multiplied with epsilon after each episode
        self.epsilon_min = epsilon_min # minimum value of epsilon 
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.alpha_decay = alpha_decay # learning rate decay factor
        self.batch_size = batch_size # number of samples used for training

        self.model = self._build_model()

    # Creating deep neural network model to output Q-values 
    def _build_model(self):
        # Sequential API used to create the neural network
        model = Sequential() # Sequential() creates the foundation of layers
        # Densely connected layers used due to simple working environment
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear')) # get Q-value for each action
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model
    
    def act(self, state, train_episodes, episode): 
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()  # Explore: choose a random action
        act_values = self.model.predict(np.reshape(state, (1, self.state_size)))
        return np.argmax(act_values[0])  # Exploit: choose the action with max Q-value

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(np.reshape(state, (1, self.state_size)))
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(np.reshape(next_state, (1, self.state_size)))[0])
            x_batch.append(np.reshape(state, (1, self.state_size))[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ID of the working environment
env_id = "CartPole-v1"

# Variables for training/testing
test_episodes = 200   # number of episodes for testing
train_episodes = 600  # number of episodes for training
episodes = train_episodes + test_episodes   # total number of episodes
max_steps = 100     # maximum number of steps per episode

# Initialize the DQN agent
agent = DQNAgent(env_id)

# Arrays to store the results
rewards = []
steps = []

# Main loop
for episode in range(episodes):
    state = agent.env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = agent.act(state, train_episodes, episode)
        next_state, reward, done, _ = agent.env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if episode < train_episodes:
        agent.replay()
    rewards.append(total_reward)
    steps.append(step + 1)

    print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {step + 1}, Epsilon: {agent.epsilon:.2f}")

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(episodes), rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')

plt.subplot(1, 2, 2)
plt.plot(range(episodes), steps)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')

plt.show()
