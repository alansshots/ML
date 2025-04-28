import numpy as np
import matplotlib.pyplot as plt


def value_iteration(grid_size, rewards, discount_factor, max_iterations):
    # Initialize utility grid
    U = np.full(grid_size, -1.0)
    # Set terminal state utility
    U[0, 2] = 10.0

    utilities_over_time = []

    for _ in range(max_iterations):
        U_new = U.copy()
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if (i, j) == (0, 2):  # Terminal state
                    continue
                values = []
                if i > 0:  # Up
                    values.append(0.8 * U[i - 1, j] + 0.1 * U[i, j - 1 if j > 0 else j] + 0.1 * U[
                        i, j + 1 if j < grid_size[1] - 1 else j])
                else:
                    values.append(0.8 * U[i, j] + 0.1 * U[i, j - 1 if j > 0 else j] + 0.1 * U[
                        i, j + 1 if j < grid_size[1] - 1 else j])
                if i < grid_size[0] - 1:  # Down
                    values.append(0.8 * U[i + 1, j] + 0.1 * U[i, j - 1 if j > 0 else j] + 0.1 * U[
                        i, j + 1 if j < grid_size[1] - 1 else j])
                else:
                    values.append(0.8 * U[i, j] + 0.1 * U[i, j - 1 if j > 0 else j] + 0.1 * U[
                        i, j + 1 if j < grid_size[1] - 1 else j])
                if j > 0:  # Left
                    values.append(0.8 * U[i, j - 1] + 0.1 * U[i - 1 if i > 0 else i, j] + 0.1 * U[
                        i + 1 if i < grid_size[0] - 1 else i, j])
                else:
                    values.append(0.8 * U[i, j] + 0.1 * U[i - 1 if i > 0 else i, j] + 0.1 * U[
                        i + 1 if i < grid_size[0] - 1 else i, j])
                if j < grid_size[1] - 1:  # Right
                    values.append(0.8 * U[i, j + 1] + 0.1 * U[i - 1 if i > 0 else i, j] + 0.1 * U[
                        i + 1 if i < grid_size[0] - 1 else i, j])
                else:
                    values.append(0.8 * U[i, j] + 0.1 * U[i - 1 if i > 0 else i, j] + 0.1 * U[
                        i + 1 if i < grid_size[0] - 1 else i, j])

                U_new[i, j] = rewards[i, j] + discount_factor * max(values)
        U = U_new
        utilities_over_time.append(U.copy())

    return U, utilities_over_time


def extract_policy(U, grid_size):
    policy = np.full(grid_size, '', dtype=object)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i, j) == (0, 2):  # Terminal state
                policy[i, j] = 'T'
                continue
            values = []
            actions = []
            if i > 0:  # Up
                values.append(0.8 * U[i - 1, j] + 0.1 * U[i, j - 1 if j > 0 else j] + 0.1 * U[
                    i, j + 1 if j < grid_size[1] - 1 else j])
                actions.append('U')
            if i < grid_size[0] - 1:  # Down
                values.append(0.8 * U[i + 1, j] + 0.1 * U[i, j - 1 if j > 0 else j] + 0.1 * U[
                    i, j + 1 if j < grid_size[1] - 1 else j])
                actions.append('D')
            if j > 0:  # Left
                values.append(0.8 * U[i, j - 1] + 0.1 * U[i - 1 if i > 0 else i, j] + 0.1 * U[
                    i + 1 if i < grid_size[0] - 1 else i, j])
                actions.append('L')
            if j < grid_size[1] - 1:  # Right
                values.append(0.8 * U[i, j + 1] + 0.1 * U[i - 1 if i > 0 else i, j] + 0.1 * U[
                    i + 1 if i < grid_size[0] - 1 else i, j])
                actions.append('R')

            best_action = actions[np.argmax(values)]
            policy[i, j] = best_action
    return policy


# Define the grid size
grid_size = (3, 3)

# Rewards for r = +3 and r = -3
rewards_plus3 = np.array([[3, -1, 10], [-1, -1, -1], [-1, -1, -1]])
rewards_minus3 = np.array([[-3, -1, 10], [-1, -1, -1], [-1, -1, -1]])

# Discount factor
gamma = 0.5

# Number of iterations for convergence
max_iterations = 30

# Calculate utilities and policies for r = +3
U_plus3, utilities_over_time_plus3 = value_iteration(grid_size, rewards_plus3, gamma, max_iterations)
policy_plus3 = extract_policy(U_plus3, grid_size)

# Calculate utilities and policies for r = -3
U_minus3, utilities_over_time_minus3 = value_iteration(grid_size, rewards_minus3, gamma, max_iterations)
policy_minus3 = extract_policy(U_minus3, grid_size)


# Plot the converged utilities for visualization
def plot_converged_utilities(utilities_over_time, title):
    plt.figure(figsize=(10, 6))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i, j) == (0, 2):
                continue
            plt.plot(range(len(utilities_over_time)), [u[i, j] for u in utilities_over_time],
                     label=f"({i + 1},{j + 1})")
    plt.xlabel('Number of iterations')
    plt.ylabel('Utility estimates')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_converged_utilities(utilities_over_time_plus3, "Converged utilities for r = +3")
plot_converged_utilities(utilities_over_time_minus3, "Converged utilities for r = -3")

# Display the results
print("Utilities for r = +3:")
print(U_plus3)
print("Policy for r = +3:")
print(policy_plus3)

print("Utilities for r = -3:")
print(U_minus3)
print("Policy for r = -3:")
print(policy_minus3)
