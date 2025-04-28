import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data

def initialize_centroids(X, k):
    """Randomly select K data points from the dataset as initial cluster centroids"""
    centroids_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[centroids_indices]
    return centroids

def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid"""
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(X, clusters, k):
    """Update centroids based on mean of data points in each cluster"""
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(X[clusters == i], axis=0)
    return centroids

def calculate_sse(X, centroids, clusters):
    """Calculate sum of squared errors (SSE)"""
    sse = 0
    for i, centroid in enumerate(centroids):
        sse += np.sum((X[clusters == i] - centroid) ** 2)
    return sse

def k_means(X, k, max_iterations=100):
    """K-Means clustering algorithm"""
    centroids = initialize_centroids(X, k)
    best_sse = float('inf')
    best_centroids = None
    best_clusters = None
    sse_history = []
    for _ in range(max_iterations):
        clusters = assign_clusters(X, centroids)
        old_centroids = centroids.copy()
        centroids = update_centroids(X, clusters, k)
        sse = calculate_sse(X, centroids, clusters)
        sse_history.append(sse)
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            best_clusters = clusters
        # Check for small changes in centroids
        if np.linalg.norm(old_centroids - centroids) < 1e-5:
            break  # If centroids change insignificantly, break out of loop
    print("Best Centroids:")
    print(best_centroids)
    print("Best Labels:")
    print(best_clusters)
    print("Sum of Squared Errors (SSE):", best_sse)
    return best_centroids, best_clusters, sse_history


# def k_means(X, k, max_iterations=100):
#     """K-Means clustering algorithm"""
#     best_sse = float('inf')
#     best_centroids = None
#     best_clusters = None
#     sse_history = []
#     for _ in range(max_iterations):
#         centroids = initialize_centroids(X, k)
#         clusters = assign_clusters(X, centroids)
#         old_centroids = centroids.copy()
#         centroids = update_centroids(X, clusters, k)
#         sse = calculate_sse(X, centroids, clusters)
#         sse_history.append(sse)
#         if sse < best_sse:
#             best_sse = sse
#             best_centroids = centroids
#             best_clusters = clusters
#     print("Best Centroids:")
#     print(best_centroids)
#     print("Best Labels:")
#     print(best_clusters)
#     print("Sum of Squared Errors (SSE):", best_sse)
#     return best_centroids, best_clusters, sse_history

# def k_means(X, k, max_iterations=100):
#     """K-Means clustering algorithm"""
#     centroids = initialize_centroids(X, k)
#     best_sse = float('inf')
#     best_centroids = None
#     best_clusters = None
#     sse_history = []
#     for _ in range(max_iterations):
#         clusters = assign_clusters(X, centroids)
#         old_centroids = centroids.copy()
#         centroids = update_centroids(X, clusters, k)
#         sse = calculate_sse(X, centroids, clusters)
#         sse_history.append(sse)
#         if sse < best_sse:
#             best_sse = sse
#             best_centroids = centroids
#             best_clusters = clusters
#         if np.all(old_centroids == centroids):
#             break  # If centroids no longer change, break out of loop
#     print("Best Centroids:")
#     print(best_centroids)
#     print("Best Labels:")
#     print(best_clusters)
#     print("Sum of Squared Errors (SSE):", best_sse)
#     return best_centroids, best_clusters, sse_history


def plot_sse_convergence(sse_history):
    """Plot SSE against number of iterations"""
    plt.plot(range(1, len(sse_history) + 1), sse_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Error Convergence')
    plt.grid(True)
    plt.show()

# Set parameters
k = 3  # Number of clusters
max_iterations = 100  # Maximum number of iterations

# Run K-Means algorithm
centroids, clusters, sse_history = k_means(X, k, max_iterations)

# Plot SSE convergence
plot_sse_convergence(sse_history)
