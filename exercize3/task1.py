import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Training function
def train(X, y, epochs, learning_rate):
    # Initialize weights and biases
    np.random.seed(21)
    theta_11_1 = np.random.randn()
    theta_12_1 = np.random.randn()
    theta_21_1 = np.random.randn()
    theta_22_1 = np.random.randn()
    b_1_2 = np.random.randn()
    b_2_2 = np.random.randn()
    theta_11_2 = np.random.randn()
    theta_21_2 = np.random.randn()
    b_1_3 = np.random.randn()

    loss_history = []

    for epoch in range(epochs):
        # Forward propagation
        H1 = sigmoid(theta_11_1 * X[:, 0] + theta_12_1 * X[:, 1] + b_1_2)
        H2 = sigmoid(theta_21_1 * X[:, 0] + theta_22_1 * X[:, 1] + b_2_2)
        O = sigmoid(theta_11_2 * H1 + theta_21_2 * H2 + b_1_3)

        # Loss computation
        loss = mean_squared_error(y, O)
        loss_history.append(loss)

        # Backpropagation
        d_loss_o = O - y
        d_loss_theta_11_2 = np.dot(H1, d_loss_o * sigmoid_derivative(O)) / X.shape[0]
        d_loss_theta_21_2 = np.dot(H2, d_loss_o * sigmoid_derivative(O)) / X.shape[0]
        d_loss_b_1_3 = np.mean(d_loss_o * sigmoid_derivative(O))

        d_loss_h1 = np.dot(d_loss_o * sigmoid_derivative(O), theta_11_2) * sigmoid_derivative(H1)
        d_loss_h2 = np.dot(d_loss_o * sigmoid_derivative(O), theta_21_2) * sigmoid_derivative(H2)
        d_loss_theta_11_1 = np.dot(X[:, 0], d_loss_h1) / X.shape[0]
        d_loss_theta_12_1 = np.dot(X[:, 1], d_loss_h1) / X.shape[0]
        d_loss_theta_21_1 = np.dot(X[:, 0], d_loss_h2) / X.shape[0]
        d_loss_theta_22_1 = np.dot(X[:, 1], d_loss_h2) / X.shape[0]
        d_loss_b_1_2 = np.mean(d_loss_h1)
        d_loss_b_2_2 = np.mean(d_loss_h2)

        # Update weights and biases
        theta_11_1 -= learning_rate * d_loss_theta_11_1
        theta_12_1 -= learning_rate * d_loss_theta_12_1
        theta_21_1 -= learning_rate * d_loss_theta_21_1
        theta_22_1 -= learning_rate * d_loss_theta_22_1
        b_1_2 -= learning_rate * d_loss_b_1_2
        b_2_2 -= learning_rate * d_loss_b_2_2
        theta_11_2 -= learning_rate * d_loss_theta_11_2
        theta_21_2 -= learning_rate * d_loss_theta_21_2
        b_1_3 -= learning_rate * d_loss_b_1_3

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return {
        'theta_11_1': theta_11_1,
        'theta_12_1': theta_12_1,
        'theta_21_1': theta_21_1,
        'theta_22_1': theta_22_1,
        'b_1_2': b_1_2,
        'b_2_2': b_2_2,
        'theta_11_2': theta_11_2,
        'theta_21_2': theta_21_2,
        'b_1_3': b_1_3,
        'loss_history': loss_history
    }

# AND gate input and output
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# XOR gate input and output
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train the network for AND gate
params_and = train(X_and, y_and, epochs=10000, learning_rate=0.1)

# Train the network for XOR gate
params_xor = train(X_xor, y_xor, epochs=10000, learning_rate=0.1)

# Testing the network
def predict(X, params):
    H1 = sigmoid(params['theta_11_1'] * X[:, 0] + params['theta_12_1'] * X[:, 1] + params['b_1_2'])
    H2 = sigmoid(params['theta_21_1'] * X[:, 0] + params['theta_22_1'] * X[:, 1] + params['b_2_2'])
    O = sigmoid(params['theta_11_2'] * H1 + params['theta_21_2'] * H2 + params['b_1_3'])
    return O >= 0.5

# Predictions for AND gate
predictions_and = predict(X_and, params_and)

print(' ')
# Output the results in the specified format for AND gate
print("AND Gate Results:")
for i, (x, pred, actual) in enumerate(zip(X_and, predictions_and, y_and)):
    print(f'Input: {x} ; Prediction: [{pred.astype(bool)}]; Actual: [{actual}]')

# Predictions for XOR gate
predictions_xor = predict(X_xor, params_xor)

# Output the results in the specified format for XOR gate
print("\nXOR Gate Results:")
for i, (x, pred, actual) in enumerate(zip(X_xor, predictions_xor, y_xor)):
    print(f'Input: {x} ; Prediction: [{pred.astype(bool)}]; Actual: [{actual}]')

# Plotting the loss convergence for AND gate
plt.plot(params_and['loss_history'])
plt.title('Loss Convergence Over Time for AND')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plotting the loss convergence for XOR gate
plt.plot(params_xor['loss_history'])
plt.title('Loss Convergence Over Time for XOR')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
