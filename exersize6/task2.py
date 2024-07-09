import numpy as np
import matplotlib.pyplot as plt

# Define the grid size
grid_size = (3, 3)
discount_factor = 0.5
num_iterations = 30  # Set the number of iterations to 30

# Define the rewards for each case
rewards = {
    '+3': 3,
    '-3': -3
}

# Define the actions (up, down, left, right)
actions = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Initialize utility grid
def initialize_utilities(grid_size, reward):
    utilities = np.full(grid_size, -1.0)
    utilities[0, 0] = reward
    utilities[2, 2] = reward
    return utilities

# Get possible next states and their probabilities
def get_next_states_and_probs(state, action, grid_size):
    i, j = state
    di, dj = action
    intended_state = (i + di, j + dj)
    perpendicular_states = [
        (i + dj, j - di),  # right turn
        (i - dj, j + di)   # left turn
    ]
    next_states_probs = [(intended_state, 0.8)]
    for next_state in perpendicular_states:
        next_states_probs.append((next_state, 0.1))
    valid_next_states_probs = []
    for next_state, prob in next_states_probs:
        ni, nj = next_state
        if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
            valid_next_states_probs.append((next_state, prob))
        else:
            valid_next_states_probs.append((state, prob))  # wall collision
    return valid_next_states_probs

# Update utility for a state
def update_utility(state, utilities, reward, discount_factor, actions, grid_size):
    max_utility = float('-inf')
    for action in actions.values():
        utility = 0
        for next_state, prob in get_next_states_and_probs(state, action, grid_size):
            ni, nj = next_state
            utility += prob * utilities[ni, nj]
        if utility > max_utility:
            max_utility = utility
    return reward + discount_factor * max_utility

# Perform value iteration and store utilities for each iteration
def value_iteration(grid_size, reward, discount_factor, num_iterations):
    utilities = initialize_utilities(grid_size, reward)
    utility_history = [utilities.copy()]

    for _ in range(num_iterations):
        new_utilities = np.copy(utilities)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if (i, j) in [(0, 0), (2, 2)]:
                    continue
                new_utilities[i, j] = update_utility((i, j), utilities, -1, discount_factor, actions, grid_size)
        utilities = new_utilities
        utility_history.append(utilities.copy())
    return utility_history

# Plot the utilities over iterations
def plot_utilities_over_iterations(utility_history, title):
    iterations = len(utility_history)
    utility_history = np.array(utility_history)
    
    plt.figure(figsize=(10, 6))
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            plt.plot(range(iterations), utility_history[:, i, j], label=f"State ({i+1},{j+1})")
    
    plt.xlabel('Number of Iterations')
    plt.ylabel('Utility Estimates')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()

# Main function to run value iteration for both cases and plot the utilities
def main():
    for reward_name, reward in rewards.items():
        print(f"\nValue Iteration for reward {reward_name} ({reward}):")
        
        # Value iteration with fixed number of iterations
        utility_history = value_iteration(grid_size, reward, discount_factor, num_iterations)
        
        # Plot the utilities over iterations
        plot_utilities_over_iterations(utility_history, f"Utilities over {num_iterations} Iterations for reward {reward_name}")

if __name__ == "__main__":
    main()
