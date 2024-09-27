import random
import pandas as pd
from gurobipy import Model, GRB, quicksum
from sklearn.model_selection import train_test_split
import numpy as np


sp_data = pd.read_csv('modelTest\small-sample.csv')
X_sp = sp_data.drop(columns=['quality']).values
y_sp = sp_data['quality'].values
X_sp_sample, _, y_sp_sample, _ = train_test_split(X_sp, y_sp, test_size=0.2, random_state=42)

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
    model.setObjective(quicksum(x[i] for i in range(num_samples) if y[i]==1) - lambda_value * V, GRB.MAXIMIZE)
    model.addConstr(
        V >= (theta - 1) * quicksum(x[i] for i in range(num_samples) if y[i]==1) + theta * quicksum(y_vars[j] for j in range(num_samples)if y[j]==0) + theta * epsilon_R,
        name="precision_constraint"
    )

    # !!!might have problem here
    for i in range(num_samples):
        if y[i] == 1:  #P
            model.addConstr(x[i] <= 1 + sum(w[k] * X[i, k] for k in range(num_features)) - c - epsilon_P, name=f"classification_positive_{i}")
        else:  #N
            model.addConstr(y_vars[i] >= sum(w[k] * X[i, k] for k in range(num_features)) - c + epsilon_N, name=f"classification_negative_{i}")
    model.optimize()

    for v in model.getVars():
        print('%s %g' % (v.VarName, v.X))

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        initial_reach = sum(1 for i in range(num_samples) if y[i] == 1)
        bc_reach = sum(x[i].X for i in range(num_samples) if x[i].X > 0.5)
        nodes = model.NodeCount
        model.write('output.lp')
        import re

# Read the LP file
        with open('output.lp', 'r') as file:
            lines = file.readlines()

        # Initialize a list to store the modified lines
        cleaned_lines = []

        # Define a regular expression to match terms with a zero coefficient
        zero_term_pattern = re.compile(r'[\+\-]?\s*0\s*\*?\s*[a-zA-Z_]\[?\d*]')

        # Process each line
        for line in lines:
            if "Maximize" in line or any(char.isdigit() for char in line):
                # Remove terms with a coefficient of 0
                cleaned_line = zero_term_pattern.sub('', line)
                
                # Clean up extra whitespace and reformat the line
                cleaned_line = re.sub(r'\s{2,}', ' ', cleaned_line).strip()

                # Append the cleaned line to the result
                cleaned_lines.append(cleaned_line + '\n')
            else:
                # If it's not part of the objective function or constraint, just add the line as is
                cleaned_lines.append(line)

        # Write the cleaned LP to a new file
        with open('cleaned_output.lp', 'w') as file:
            file.writelines(cleaned_lines)

        print("Cleaned LP file created as 'cleaned_output.lp'")





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
results.append(wide_reach_classification(X_sp_sample, y_sp_sample, "test", theta=0.9))

df_results = pd.DataFrame(results)
print("Summary of Results:")
print(df_results.to_string(index=False))