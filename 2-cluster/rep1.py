import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

def generate_2_cluster_data(n=1000, d=8,d_max=8):
    negative_samples = np.random.rand(4 * n // 5, d)
    s = (np.math.factorial(d) / np.math.factorial(d_max)) ** (1 / d)  # side length actually here is always close to 1
    positive_cluster_1 = np.random.rand(n // 20, d)
    positive_cluster_1 = positive_cluster_1 / positive_cluster_1.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = np.random.rand(n // 20, d)
    positive_cluster_2 = positive_cluster_2 / positive_cluster_2.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = s - positive_cluster_2
    positive_samples = np.vstack((positive_cluster_1, positive_cluster_2))
    X = np.vstack((positive_samples, negative_samples))
    y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
    return X, y

def wide_reach_classification(X, y, theta=0.9, epsilon_p=0.01, epsilon_n=0.01, epsilon_r=0.01):
    model = LpProblem("WideReachClassification", LpMaximize)
    n_samples, n_features = X.shape
    xs = LpVariable.dicts("x", range(n_samples), 0, 1, LpBinary)
    ys = LpVariable.dicts("y", range(n_samples), 0, 1, LpBinary)
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
            model += lpSum([X[i, j] * w[j] for j in range(n_features)]) - c - epsilon_n >= 1 - ys[i]
    model.solve()
    
    hyperplane = [w[j].varValue for j in range(n_features)], c.varValue
    reach = sum([xs[i].varValue for i in range(n_samples) if y[i] == 1])
    precision = sum([xs[i].varValue for i in range(n_samples) if y[i] == 1]) / (sum([xs[i].varValue for i in range(n_samples)]) + epsilon_r)
    
    return hyperplane, reach, precision

X, y = generate_2_cluster_data()
hyperplane, reach, precision = wide_reach_classification(X, y)
print("Hyperplane:", hyperplane)
print("Reach:", reach)
print("Precision:", precision)


import matplotlib.pyplot as plt
def plot_clusters(X, y, hyperplane):
    positive = X[y == 1]
    negative = X[y == 0]
    
    plt.scatter(positive[:, 0], positive[:, 1], color='red', label='Positive',marker='+')
    plt.scatter(negative[:, 0], negative[:, 1], color='blue', label='Negative',marker='x')
    
    if X.shape[1] == 2:
        w, c = hyperplane
        x_vals = np.linspace(0, 1, 100)
        y_vals = (c - w[0] * x_vals) / w[1]
        plt.plot(x_vals, y_vals, color='black', label='Hyperplane')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

plot_clusters(X, y, hyperplane)

