import numpy as np
from gurobipy import Model, GRB, quicksum, LinExpr
import matplotlib.pyplot as plt

def generate_2_cluster_data(n=1000, d=2,d_max=8):
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

def compute_lambda(n, theta):
    return 10 * (n + 1) * theta

# initial num for Gurobi 
def generate_initial_solution(num_samples):
    return [0.5 for _ in range(num_samples)]

def wide_reach_classification(X, y, dataset_name, theta, epsilon_R=0.01, epsilon_P=0.01, epsilon_N=0.01):
    # get lambda
    lambda_value = compute_lambda(len(y), theta)

    model = Model("Wide-Reach_Classification")
    model.setParam(GRB.Param.TimeLimit, 120) # time
    model.setParam(GRB.Param.MIPGap, 0.01) # gap
    model.setParam(GRB.Param.Heuristics, 0)
    model.setParam(GRB.Param.NodeMethod, 2)  

    num_features = X.shape[1]
    num_samples = len(y)
    w = model.addVars(num_features, vtype=GRB.CONTINUOUS, name="w")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    x = model.addVars(num_samples, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(num_samples, vtype=GRB.BINARY, name="y")
    V = model.addVar(vtype=GRB.CONTINUOUS, name="V")

    # Apply  Lagrangian
    model.setObjective(quicksum((x[i] if y[i] ==1 else 0) for i in range(num_samples)) - lambda_value * V, GRB.MAXIMIZE)
    model.addConstr(
        V >= (theta - 1) * quicksum((x[i] if y[i] ==1 else 0) for i in range(num_samples)) + theta * quicksum((y_vars[j] if y[j] == 0 else 0)for j in range(num_samples)) + theta * epsilon_R,
        name="precision_constraint"
    )

    # !!!might have problem here
    for i in range(num_samples):
        if y[i] == 1:  #P
            model.addConstr(x[i] <= 1 + sum(w[k] * X[i, k] for k in range(num_features)) - c - epsilon_P, name=f"classification_positive_{i}")
        else:  #N
            model.addConstr(y_vars[i] >= sum(w[k] * X[i, k] for k in range(num_features)) - c + epsilon_N, name=f"classification_negative_{i}")

    model.optimize()
    print(w)
    
    # hyperplane = [w[j] for j in range(num_features)], c
    # reach = sum(1 for i in range(num_samples) if y[i] == 1)
    # divisor = sum(1 for i in range(num_samples)) + epsilon_R
    # precision = reach / divisor

    # print(hyperplane)

    # return hyperplane, reach, precision
    return 0,0,0

def plot_clusters(X, y, hyperplane):
    positive = X[y == 1]
    negative = X[y == 0]
    
    plt.scatter(positive[:, 0], positive[:, 1], color='red', label='Positive',marker='+')
    plt.scatter(negative[:, 0], negative[:, 1], color='blue', label='Negative',marker='x')
    
    if X.shape[1] == 2:
        w, c = hyperplane
        x_vals = np.linspace(0, 1, 100)
        y_vals = (c - w[0].X * x_vals) / w[1].X
        plt.plot(x_vals, y_vals, color='black', label='Hyperplane')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


X, y = generate_2_cluster_data()
hyperplane, reach, precision = wide_reach_classification(X, y, "2-cluster", theta=0.99)
# print("Hyperplane:", hyperplane)
# print("Reach:", reach)
# print("Precision:", precision)
# plot_clusters(X, y, hyperplane)



plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Negative Samples')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Positive Samples')
plt.legend()
plt.title('2-Cluster Benchmark Data Distribution')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()