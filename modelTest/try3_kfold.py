import random
import pandas as pd
from gurobipy import Model, GRB, quicksum
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

red_wine_path = 'winequality-red-corrected.csv'
white_wine_path = 'winequality-white-corrected.csv'
red_wine_data = pd.read_csv(red_wine_path)
white_wine_data = pd.read_csv(white_wine_path)
crop_data = pd.read_csv('WinnipegDataset.csv')
bc_data = pd.read_csv('wdbc.csv')

red_wine_data['target'] = (red_wine_data['quality'] >= 7).astype(int)
X_red = red_wine_data.drop(columns=['quality', 'target']).values
y_red = red_wine_data['target'].values
white_wine_data['target'] = (white_wine_data['quality'] >= 7).astype(int)
X_white = white_wine_data.drop(columns=['quality', 'target']).values
y_white = white_wine_data['target'].values

bc_data['target'] = (bc_data['diagnosis'] == 'M').astype(int)
X_bc = bc_data.drop(columns=['diagnosis', 'target']).values
y_bc = bc_data['target'].values

crop_random_sample_list = random.sample(range(325835), 10000)
crop_data = crop_data.iloc[crop_random_sample_list]
crop_data['target'] = (crop_data['label'] == 6).astype(int)
X_crop = crop_data.drop(columns=['label', 'target']).values
y_crop = crop_data['target'].values

scaler = StandardScaler()
X_red = scaler.fit_transform(X_red)
X_white = scaler.fit_transform(X_white)
X_crop = scaler.fit_transform(X_crop)
X_bc = scaler.fit_transform(X_bc)

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=14)

# adjust Lambda
def compute_lambda(n, theta):
    return 10 * (n + 1) * theta

# get intial value
def generate_initial_solution(num_samples):
    return [0.5 for _ in range(num_samples)]

def wide_reach_classification(X, y, dataset_name, theta, epsilon_R=0.01, epsilon_P=0.01, epsilon_N=0.01):
    lambda_value = compute_lambda(len(y), theta)

    model = Model("Wide-Reach_Classification")
    model.setParam(GRB.Param.TimeLimit, 180)  # time limit
    model.setParam(GRB.Param.MIPGap, 0.005)  # gap
    model.setParam(GRB.Param.Heuristics, 0.8) 
    model.setParam(GRB.Param.NodeMethod, 2) 

    num_features = X.shape[1]
    num_samples = len(y)
    w = model.addVars(num_features, vtype=GRB.CONTINUOUS, name="w")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    x = model.addVars(num_samples, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(num_samples, vtype=GRB.BINARY, name="y")
    V = model.addVar(vtype=GRB.CONTINUOUS, name="V")

    model.setObjective(quicksum(x[i] for i in range(num_samples)) - lambda_value * V, GRB.MAXIMIZE)
    model.addConstr(
        V >= (theta - 1) * quicksum(x[i] for i in range(num_samples)) + theta * quicksum(y_vars[j] for j in range(num_samples)) + theta * epsilon_R,
        name="precision_constraint"
    )

    for i in range(num_samples):
        if y[i] == 1:  #P
            model.addConstr(x[i] <= 1 + sum(w[k] * X[i, k] for k in range(num_features)) - c - epsilon_P, name=f"classification_positive_{i}")
        else:  #N
            model.addConstr(y_vars[i] >= sum(w[k] * X[i, k] for k in range(num_features)) - c + epsilon_N, name=f"classification_negative_{i}")

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        initial_reach = sum(1 for i in range(num_samples) if y[i] == 1)
        bc_reach = sum(x[i].X for i in range(num_samples) if x[i].X > 0.5)
        nodes = model.NodeCount

        print(f"{dataset_name} Dataset Results:")
        print(f"Initial Reach: {initial_reach}")
        print(f"BC Reach: {bc_reach}")
        print(f"Nodes: {nodes}\n")

        return {
            'Name': dataset_name,
            'Initial Reach': initial_reach,
            'BC Reach': bc_reach,
            'Nodes': nodes
        }
    else:
        print(f"No feasible solution found for {dataset_name}.")
        return {
            'Name': dataset_name,
            'Initial Reach': 0,
            'BC Reach': 0,
            'Nodes': 0
        }

results = []

# KFold training
for train_index, test_index in kf.split(X_bc):
    X_train, X_test = X_bc[train_index], X_bc[test_index]
    y_train, y_test = y_bc[train_index], y_bc[test_index]
    results.append(wide_reach_classification(X_train, y_train, "B&C", theta=0.99))

for train_index, test_index in kf.split(X_red):
    X_train, X_test = X_red[train_index], X_red[test_index]
    y_train, y_test = y_red[train_index], y_red[test_index]
    results.append(wide_reach_classification(X_train, y_train, "Wine Quality (red)", theta=0.04))

for train_index, test_index in kf.split(X_white):
    X_train, X_test = X_white[train_index], X_white[test_index]
    y_train, y_test = y_white[train_index], y_white[test_index]
    results.append(wide_reach_classification(X_train, y_train, "Wine Quality (white)", theta=0.1))

for train_index, test_index in kf.split(X_crop):
    X_train, X_test = X_crop[train_index], X_crop[test_index]
    y_train, y_test = y_crop[train_index], y_crop[test_index]
    results.append(wide_reach_classification(X_train, y_train, "crop", theta=0.99))

df_results = pd.DataFrame(results)
print("Summary of Results:")
print(df_results.to_string(index=False))
