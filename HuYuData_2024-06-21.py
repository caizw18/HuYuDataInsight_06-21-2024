import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split features
X = df.drop(columns='target').values

# Define helper functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def initialize_centroids(X, k):
    np.random.seed(42)  # For reproducibility
    random_idx = np.random.permutation(X.shape[0])
    centroids = X[random_idx[:k]]
    return centroids

# Define the K-means classifier
class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        self.centroids = initialize_centroids(X, self.k)

        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            old_centroids = self.centroids
            self.centroids = self._update_centroids(X)

            if self._is_converged(old_centroids, self.centroids):
                break

    def _assign_clusters(self, X):
        labels = []
        for x in X:
            distances = [euclidean_distance(x, centroid) for centroid in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def _update_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        for idx in range(self.k):
            cluster_points = X[self.labels == idx]
            centroids[idx] = cluster_points.mean(axis=0)
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        distances = [euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.k)]
        return np.sum(distances) < self.tol

    def predict(self, X):
        return self._assign_clusters(X)

# Initialize and train the KMeans classifier
kmeans = KMeans(k=3, max_iters=100, tol=1e-4)
kmeans.fit(X)

# Make predictions
y_pred = kmeans.predict(X)

# Evaluate the clustering (using silhouette score as an example)
from sklearn.metrics import silhouette_score
score = silhouette_score(X, y_pred)

print(f'Silhouette Score: {score}')
print(f'Centroids:\n{kmeans.centroids}')