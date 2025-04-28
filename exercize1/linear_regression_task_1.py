# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Task 1

# Dataset Generation:
# Random x values
X = np.random.rand(100, 1)
# y values with some noise
y = 2 * X + 3 + np.random.randn(100, 1)

# Data Splitting:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Implementation
class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y, lr=0.01, iterations=1000):
        # Initialize coefficients
        self.theta = np.zeros((2, 1))
        m = len(y)

        for _ in range(iterations):
            # Calculate predictions
            y_pred = np.dot(X, self.theta)

            # Update coefficients
            self.theta -= lr * (1/m) * np.dot(X.T, y_pred - y)

    def predict(self, X):
        return np.dot(X, self.theta)

# Model Training
# Add bias term to X_train
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]

# Train the model
lr = LinearRegression()
lr.fit(X_train_b, y_train)

# Model evaluation
# Add bias term to X_test
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# Predict
y_pred = lr.predict(X_test_b)

# Calculate MSE
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)

# Model Visualization
# Plotting the training data
plt.scatter(X_train, y_train, color='blue')
# Plotting the regression line
plt.plot(X_train, lr.predict(X_train_b), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression(Line)/Training Data(Blue)')
plt.show()
