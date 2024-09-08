import pandas as pd
from gurobipy import Model, GRB, quicksum
from sklearn.model_selection import train_test_split

red_wine_path = 'winequality-red-corrected.csv'
white_wine_path = 'winequality-white-corrected.csv'
red_wine_data = pd.read_csv(red_wine_path)
white_wine_data = pd.read_csv(white_wine_path)

red_wine_data['target'] = (red_wine_data['quality'] >= 7).astype(int)
X_red = red_wine_data.drop(columns=['quality', 'target']).values
y_red = red_wine_data['target'].values
white_wine_data['target'] = (white_wine_data['quality'] >= 7).astype(int)
X_white = white_wine_data.drop(columns=['quality', 'target']).values
y_white = white_wine_data['target'].values

# randon select(50%)
X_red_sample, _, y_red_sample, _ = train_test_split(X_red, y_red, test_size=0.5, random_state=42)
X_white_sample, _, y_white_sample, _ = train_test_split(X_white, y_white, test_size=0.5, random_state=42)

def wide_reach_classification(X, y, dataset_name, theta=0.9, epsilon_R=0.01, epsilon_P=0.01, epsilon_N=0.01, lambda_value=10):
    model = Model("Wide-Reach_Classification")

    # limit time
    model.setParam(GRB.Param.TimeLimit, 180)

    # variable
    num_features = X.shape[1]
    num_samples = len(y)
    w = model.addVars(num_features, vtype=GRB.CONTINUOUS, name="w")
    c = model.addVar(vtype=GRB.CONTINUOUS, name="c")
    x = model.addVars(num_samples, vtype=GRB.BINARY, name="x")
    y_vars = model.addVars(num_samples, vtype=GRB.BINARY, name="y")
    V = model.addVar(vtype=GRB.CONTINUOUS, name="V")

    model.setObjective(quicksum(x[i] for i in range(num_samples)) - lambda_value * V, GRB.MAXIMIZE)

    # fomula
    model.addConstr(
        V >= (theta - 1) * quicksum(x[i] for i in range(num_samples)) + theta * quicksum(y_vars[j] for j in range(num_samples)) + theta * epsilon_R,
        name="precision_constraint"
    )

    for i in range(num_samples):
        if y[i] == 1:  # positive
            model.addConstr(x[i] <= 1 + sum(w[k] * X[i, k] for k in range(num_features)) - c - epsilon_P, name=f"classification_positive_{i}")
        else:  # negative
            model.addConstr(y_vars[i] >= sum(w[k] * X[i, k] for k in range(num_features)) - c + epsilon_N, name=f"classification_negative_{i}")

    model.optimize()
    initial_reach = sum(1 for i in range(num_samples) if y[i] == 1)
    bc_reach = sum(x[i].X for i in range(num_samples) if x[i].X > 0.5) if model.status == GRB.OPTIMAL else 0
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

results = []
results.append(wide_reach_classification(X_red_sample, y_red_sample, "Wine Quality (red)"))
results.append(wide_reach_classification(X_white_sample, y_white_sample, "Wine Quality (white)"))
df_results = pd.DataFrame(results)
print("Summary of Results:")
print(df_results.to_string(index=False))
