import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# Define the dataset

'''
---- Students --> make it with m=20 samples
'''

# np.random.seed(1)
# X = np.random.randn(20, 2)
# y = np.array([1] * 10 + [-1] * 10)
# X[:10] += 1  # Offset the first 10 samples to make classes separable

X = np.array([
    [2.0, 3.0],
    [1.0, 1.0],
    [2.5, 1.5],
    [3.0, 2.0],
    [2.0, 0.5],
    [3.5, 1.0],
    [1.5, 2.5],
    [3.0, 3.5],
    [1.0, 0.0],
    [0.5, 2.0],
    [2.0, 4.0],
    [3.0, 1.0],
    [1.0, 3.0],
    [2.5, 2.5],
    [3.5, 0.5],
    [0.0, 1.5],
    [2.0, 1.0],
    [1.5, 3.0],
    [2.5, 3.0],
    [3.0, 0.5]
])
y = np.array([1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1])


def linear_svm(X, y):
    m, n = X.shape  # Number of samples and features
    y = y.astype(float)  # Ensure y is float type

    # Compute the kernel matrix
    K = np.dot(X, X.T)

    # Construct the matrices for the QP problem using cvxopt
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones(m))
    A = cvxopt.matrix(y, (1, m), tc='d')
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.diag(-np.ones(m)))
    h = cvxopt.matrix(np.zeros(m))

    # Solve the QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])


    # Support vectors have non-zero lagrange multipliers
    '''
    ---- Students: Write the SVM related code here ----
    1. Compute the weight vector w
    2. Compute the intercept b
    '''
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    alphas = alphas[sv]
    sv_X = X[sv]
    sv_y = y[sv]

    # Compute the weight vector w
    w = np.zeros(n)
    for n in range(len(alphas)):
        w += alphas[n] * sv_y[n] * sv_X[n]

    # Compute the intercept b
    b = 0
    for n in range(len(alphas)):
        b += sv_y[n] - np.dot(w, sv_X[n])
    b /= len(alphas)



    # -------------------------------------

    return w, b, alphas, sv


# Train the SVM (complete the returned values once you write the code)
w, b, alphas, sv = linear_svm(X, y)


def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)


def calculate_error(X, y, w, b):
    misclassified = 0

    # Predict the classes
    predictions = predict(X, w, b)

    # Check misclassifications
    for i in range(len(y)):
        if predictions[i] != y[i]:
            misclassified += 1

    error = misclassified / len(y)
    return error


# Calculate the misclassification error
error = calculate_error(X, y, w, b)
print(f'Misclassification error: {error * 100}%')


# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

'''
-----Students: plot the decision boundary once SVM implementation is complete
'''
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVM with Hard Margin')

# Plot the decision boundary
x1_plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_plot = -(w[0] * x1_plot + b) / w[1]
plt.plot(x1_plot, x2_plot, 'k-')

# Highlight the support vectors

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVM with Hard Margin')
plt.show()
