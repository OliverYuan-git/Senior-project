import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from gurobipy import Model, GRB, quicksum

def generate_custom_2_cluster_data(n_negative=320, n_positive=80, d=2):
    negative_samples = np.random.rand(n_negative, d)
    positive_cluster_1 = np.random.rand(n_positive // 2, d) * 0.2  # around 0
    positive_cluster_2 = np.random.rand(n_positive // 2, d)  
    positive_samples = np.vstack((positive_cluster_1, positive_cluster_2))
    X = np.vstack((positive_samples, negative_samples))
    y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
    return X, y

# def generate_custom_2_cluster_data(n=400, d=8, d_max=8):
#     negative_samples = np.random.rand(4 * n // 5, d)
#     s = (np.math.factorial(d) / np.math.factorial(d_max)) ** (1 / d)  
#     positive_uniform = np.random.rand(n // 10, d)
#     positive_cluster_1 = np.random.rand(n // 20, d)
#     positive_cluster_1 = positive_cluster_1 / positive_cluster_1.sum(axis=1, keepdims=True) * s
#     positive_cluster_2 = np.random.rand(n // 20, d)
#     positive_cluster_2 = positive_cluster_2 / positive_cluster_2.sum(axis=1, keepdims=True) * s
#     positive_cluster_2 = s - positive_cluster_2
#     positive_samples = np.vstack((positive_uniform, positive_cluster_1, positive_cluster_2))
#     X = np.vstack((positive_samples, negative_samples))
#     y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
#     return X, y

X, y = generate_custom_2_cluster_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Negative Samples',marker='+')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Positive Samples',marker='x')
plt.legend()
plt.title('2-Cluster Benchmark Data Distribution')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()

# WRC
def compute_lambda(n, theta):
    return 10 * (n + 1) * theta

def generate_initial_solution(num_samples):
    return [0.5 for _ in range(num_samples)]

def wide_reach_classification(X, y, theta, epsilon_R=0.01, epsilon_P=0.01, epsilon_N=0.01):
    lambda_value = compute_lambda(len(y), theta)
    model = Model("Wide-Reach_Classification")
    model.setParam(GRB.Param.TimeLimit, 500)  
    model.setParam(GRB.Param.MIPGap, 0.01)   
    model.setParam(GRB.Param.Heuristics, 0)
    model.setParam(GRB.Param.NodeMethod, 2)
    num_features = X.shape[1]
    num_samples = len(y)
    w = model.addVars(num_features, vtype=GRB.CONTINUOUS, name="w")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    x = model.addVars(num_samples, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(num_samples, vtype=GRB.BINARY, name="y")
    V = model.addVar(vtype=GRB.CONTINUOUS, name="V")

    model.setObjective(
        quicksum((x[i] if y[i] == 1 else 0) for i in range(num_samples)) - lambda_value * V,
        GRB.MAXIMIZE
    )
    
    model.addConstr(
        V >= (theta - 1) * quicksum((x[i] if y[i] == 1 else 0) for i in range(num_samples)) +
        theta * quicksum((y_vars[j] if y[j] == 0 else 0) for j in range(num_samples)) + theta * epsilon_R,
        name="precision_constraint"
    )

    for i in range(num_samples):
        if y[i] == 1:  # P
            model.addConstr(x[i] <= 1 + quicksum(w[k] * X[i, k] for k in range(num_features)) - c - epsilon_P, name=f"classification_positive_{i}")
        else:  # N
            model.addConstr(y_vars[i] >= quicksum(w[k] * X[i, k] for k in range(num_features)) - c + epsilon_N, name=f"classification_negative_{i}")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution_x = model.getAttr('x', x)
        solution_y_vars = model.getAttr('x', y_vars)
        return solution_x, solution_y_vars, w, c
    else:
        print("Model did not solve to optimality or time limit exceeded.")
        return None, None, None, None

theta = 0.9  
solution_x, solution_y_vars, w, c = wide_reach_classification(X_train, y_train, theta=theta)

if solution_x is not None and solution_y_vars is not None:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='purple', label='Positive Samples', marker= '+')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='cyan', label='Negative Samples', marker= 'x')
    plt.legend()
    plt.title('Wide-Reach Classification Results on Custom 2-Cluster Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()
else:
    print("Failed to visualize due to optimization issues.")
