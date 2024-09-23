import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

def generate_2_cluster_data(n=1000, d=8, d_max=8):
    # Calculate the side length of the simplex
    s = (np.math.factorial(d) / np.math.factorial(d_max)) ** (1 / d)
    
    # Generate negative samples uniformly in [0, 1]^d
    negative_samples = np.random.rand(4 * n // 5, d)
    
    # Generate positive samples in the simplex C
    positive_cluster_1 = np.random.rand(n // 20, d)
    positive_cluster_1 = positive_cluster_1 / positive_cluster_1.sum(axis=1, keepdims=True) * s

    # Generate positive samples in the opposite simplex C'
    positive_cluster_2 = np.random.rand(n // 20, d)
    positive_cluster_2 = positive_cluster_2 / positive_cluster_2.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = s - positive_cluster_2
    
    # Combine positive samples
    positive_samples = np.vstack((positive_cluster_1, positive_cluster_2))

    # Combine positive and negative samples
    X = np.vstack((positive_samples, negative_samples))
    y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
    return X, y

def wide_reach_classification(X, y, theta=0.9, epsilon_p=0.01, epsilon_n=0.01, epsilon_r=0.01):
    model = LpProblem("WideReachClassification", LpMaximize)
    n_samples, n_features = X.shape
    xs = LpVariable.dicts("x", range(n_samples), 0, 1, LpBinary)
    w = LpVariable.dicts("w", range(n_features))
    c = LpVariable("c")
    
    # Maximize reach
    model += lpSum([xs[i] for i in range(n_samples) if y[i] == 1])
    
    # Precision constraint
    model += lpSum([xs[i] for i in range(n_samples) if y[i] == 1]) >= theta * (lpSum([xs[i] for i in range(n_samples)]) + epsilon_r)
    
    # Classification constraints
    for i in range(n_samples):
        if y[i] == 1:
            model += lpSum([X[i, j] * w[j] for j in range(n_features)]) - c + epsilon_p <= xs[i]
        else:
            model += lpSum([X[i, j] * w[j] for j in range(n_features)]) - c - epsilon_n >= 1 - xs[i]
    model.solve()
    
    # Extract results
    hyperplane = [w[j].varValue for j in range(n_features)], c.varValue
    reach = sum([xs[i].varValue for i in range(n_samples) if y[i] == 1])
    precision = sum([xs[i].varValue for i in range(n_samples) if y[i] == 1]) / (sum([xs[i].varValue for i in range(n_samples)]) + epsilon_r)
    
    return hyperplane, reach, precision

X, y = generate_2_cluster_data(n=1000, d=8, d_max=8)
hyperplane, reach, precision = wide_reach_classification(X, y)
print("Hyperplane:", hyperplane)
print("Reach:", reach)
print("Precision:", precision)

import matplotlib.pyplot as plt

def plot_clusters(X, y, hyperplane):
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    
    positive = X[y == 1]
    negative = X[y == 0]
    
    plt.scatter(positive[:, 0], positive[:, 1], color='red', label='Positive', marker='+')
    plt.scatter(negative[:, 0], negative[:, 1], color='blue', label='Negative', marker='x')
    
    if X.shape[1] == 2:
        w, c = hyperplane
        if len(w) == 2:
            x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            y_vals = (c - w[0] * x_vals) / w[1]
            plt.plot(x_vals, y_vals, color='black', label='Hyperplane')
    
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Visualization with Hyperplane')
    plt.show()

plot_clusters(X, y, hyperplane)
