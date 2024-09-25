import numpy as np
import matplotlib.pyplot as plt

def generate_cluster_data(n=1000, d=8, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)
    negative_samples = np.random.rand(4 * n // 5, d)
    #postive
    s = 1
    positive_cluster_1 = np.random.rand(n // 20, d) * s
    positive_cluster_2 = np.random.rand(n // 20, d)
    positive_samples = np.vstack((positive_cluster_1, positive_cluster_2))
    X = np.vstack((positive_samples, negative_samples))
    y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
    return X, y

X, y = generate_cluster_data(n=1000, d=8, random_seed=42)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Positive', marker='x')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Negative', marker='+')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Cluster Benchmark Data')
plt.show()


from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

def wide_reach_classification(X, y, theta=0.90, epsilon_p=0.01, epsilon_n=0.01, epsilon_r=0.01):

    model = LpProblem("WideReachClassification", LpMaximize)
    n_samples, n_features = X.shape
    xs = LpVariable.dicts("x", range(n_samples), 0, 1, LpBinary)
    ys = LpVariable.dicts("y", range(n_samples), 0, 1, LpBinary)
    w = LpVariable.dicts("w", range(n_features))
    c = LpVariable("c")
    model += lpSum([xs[i] for i in range(n_samples) if y[i] == 1])
    model += lpSum([xs[i] for i in range(n_samples) if y[i] == 1]) >= theta * (lpSum([xs[i] for i in range(n_samples)]) + epsilon_r)

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

hyperplane, reach, precision = wide_reach_classification(X, y)
print("Hyperplane:", hyperplane)
print("Reach:", reach)
print("Precision:", precision)
