import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate Dataset
np.random.seed(1)
x = 2 * np.random.rand(100, 3)
a = np.array([1.5, -2.0, 1.0]).reshape(-1, 1)
b = 0.5
noise = np.random.uniform(-1, 1, (100, 1))
y = x @ a + b + noise


# Splitting the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=37)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Model implementation
class MultivariableLinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, x, y, lr=0.01, iterations = 1000):
        m, n = x.shape
        x_b = np.c_[np.ones((m, 1)), x]
        self.theta = np.zeros((n + 1, 1))
        self.cost_history = []


        for _ in range(iterations):
            y_pred = x_b @ self.theta
            gradients = 2 / m * x_b.T @ (y_pred - y)
            self.theta -= lr * gradients
            self.cost_history.append(np.mean((y_pred - y) ** 2))

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b @ self.theta        

# Model Training and Evaluation
lrs = [0.001, 0.005, 0.01, 0.09]  # learning rates
plt.figure(figsize=(10, 8))

for i, lr in enumerate(lrs):
    model = MultivariableLinearRegression()
    model.fit(x_train, y_train, lr, iterations=500)
    plt.plot(model.cost_history, label=f'LR={lr}')

    # Evaluate the model
    y_pred = model.predict(x_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Mean Squared Error with LR={lr}: {mse:.4f}")

# Visualization
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss by Learning Rate')
plt.legend()
plt.show()
