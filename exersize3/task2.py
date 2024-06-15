import numpy as np
import matplotlib.pyplot as plt

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of the relu
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# MSE
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate data points for x and find y
np.random.seed(21)
x = np.linspace(-1, 1, 100).reshape(1, -1)
y = (3 * x**5 + 1.5 * x**4 + 2 * x**3 + 7 * x + 0.5).reshape(1, -1)

# weights and biases
def initialize_parameters():
    np.random.seed(42)
    params = {
        'W1': np.random.randn(8, 1),
        'b1': np.random.randn(8, 1),
        'W2': np.random.randn(8, 8),
        'b2': np.random.randn(8, 1),
        'W3': np.random.randn(1, 8),
        'b3': np.random.randn(1, 1)
    }
    return params

# Forward propagation
def forward_propagation(X, params):
    Z1 = np.dot(params['W1'], X) + params['b1']
    A1 = relu(Z1)
    Z2 = np.dot(params['W2'], A1) + params['b2']
    A2 = relu(Z2)
    Z3 = np.dot(params['W3'], A2) + params['b3']
    A3 = Z3  # Identity function for the output layer
    return Z1, A1, Z2, A2, Z3, A3

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, params):
    m = X.shape[1]
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    dA2 = np.dot(params['W3'].T, dZ3)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(params['W2'].T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
        'dW3': dW3,
        'db3': db3
    }
    
    return grads

# Update parameters
def update_parameters(params, grads, learning_rate):
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    params['W3'] -= learning_rate * grads['dW3']
    params['b3'] -= learning_rate * grads['db3']
    return params

# Training function
def train(X, Y, epochs, learning_rate):
    params = initialize_parameters()
    loss_history = []
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, params)
        loss = mean_squared_error(Y, A3)
        loss_history.append(loss)
        
        grads = backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, params)
        params = update_parameters(params, grads, learning_rate)
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return params, loss_history

X = x
Y = y

# Train the neural network
epochs = 10000
learning_rate = 0.01
params, loss_history = train(X, Y, epochs, learning_rate)

# Plotting the loss convergence
plt.plot(loss_history)
plt.title('Loss Convergence Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Predict function
def predict(X, params):
    Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, params)
    return A3

# predicted values vs actual values
predictions = predict(X, params)

# Plot
plt.scatter(x.flatten(), y.flatten(), color='blue', label='Actual')
plt.scatter(x.flatten(), predictions.flatten(), color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
