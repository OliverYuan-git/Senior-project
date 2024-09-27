import numpy as np
import matplotlib.pyplot as plt

def generate_2_cluster_data(n=400, d=8, d_max=8):
    negative_samples = np.random.rand(4 * n // 5, d)
    s = (np.math.factorial(d) / np.math.factorial(d_max)) ** (1 / d)  
    positive_uniform = np.random.rand(n // 10, d)
    positive_cluster_1 = np.random.rand(n // 20, d)
    positive_cluster_1 = positive_cluster_1 / positive_cluster_1.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = np.random.rand(n // 20, d)
    positive_cluster_2 = positive_cluster_2 / positive_cluster_2.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = s - positive_cluster_2
    positive_samples = np.vstack((positive_uniform, positive_cluster_1, positive_cluster_2))
    X = np.vstack((positive_samples, negative_samples))
    y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
    return X, y

X, y = generate_2_cluster_data()

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Negative Samples')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Positive Samples')
plt.legend()
plt.title('2-Cluster Benchmark Data Distribution')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()