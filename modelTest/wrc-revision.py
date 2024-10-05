import random
import pandas as pd
from gurobipy import Model, GRB, quicksum
from sklearn.model_selection import train_test_split
import numpy as np
import re

red_wine = 'winequality-red-corrected.csv'
white_wine = 'winequality-white-corrected.csv'
#crop = 'WinnipegDataset.csv'
#bc = 'wdbc.csv'

def compute_lambda(n, theta):
    return 10 * (n + 1) * theta

# initial num for Gurobi 
def generate_initial_solution(num_samples):
    return [0.5 for _ in range(num_samples)]

def clean_output(file):
        with open(file, 'r') as f:
            lines = f.readlines()

        # Initialize a list to store the modified lines
        cleaned_lines = []

        # Define a regular expression to match terms with a zero coefficient
        zero_term_pattern = re.compile(r'[\+\-]?\s*0\s*\*?\s*[a-zA-Z_]\[?\d*]')

        # Flag to indicate whether we are in the Maximize section
        in_maximize_section = False

        # Process each line
        for line in lines:
            # Check if we are in the Maximize section
            if "Maximize" in line:
                in_maximize_section = True
                cleaned_lines.append(line.strip() + '\n')
                continue

            if in_maximize_section:
                # If we reach a non-empty constraint line, we leave the Maximize section
                if "Subject To" in line or line.strip() == "":
                    in_maximize_section = False
                    cleaned_lines.append('\n' + line.strip() + '\n')
                    continue

                # Remove terms with a coefficient of 0
                cleaned_line = zero_term_pattern.sub('', line)

                # Remove excessive spaces and newlines between terms
                cleaned_line = cleaned_line.strip()

                # If there are terms on this line, append it to the Maximize section without extra newlines
                if cleaned_line:
                    cleaned_lines.append(cleaned_line + ' ')

            else:
                # If it's not part of the Maximize section, just add the line as is
                cleaned_lines.append(line)

        # Join the cleaned lines and ensure proper formatting with newlines where necessary
        cleaned_content = ''.join(cleaned_lines).strip() + '\n'

        # Write the cleaned LP to a new file
        with open(f'{file}_cleaned.lp', 'w') as f:
            f.write(cleaned_content)

def data_preprocessing(dataset_path):
    if dataset_path == red_wine:
        dataset_name = 'red-wine'
        data = pd.read_csv(red_wine)
        data['target'] = (data['quality'] >= 8).astype(int)
        X = data.drop(columns=['quality', 'target']).values
        y = data['target'].values
        return dataset_name,X,y
    elif dataset_path == white_wine:
        dataset_name = 'white-wine'
        data = pd.read_csv(white_wine)
        data['target'] = (data['quality'] >= 8).astype(int)
        X = data.drop(columns=['quality', 'target']).values
        y = data['target'].values
        return dataset_name,X,y
    elif dataset_path == crop:
        dataset_name = 'crop'
        data = pd.read_csv(crop)
        crop_random_sample_list = random.sample(range(325835), 10000)
        data = data.iloc[crop_random_sample_list]
        data['target'] = (data['label'] == 6).astype(int)
        X = data.drop(columns=['label', 'target']).values
        y = data['target'].values
        return dataset_name,X,y
    elif dataset_path == bc:
        dataset_name = 'B&C'
        data = pd.read_csv(bc)
        data['target'] = (data['diagnosis'] == 'M').astype(int)
        X = data.drop(columns=['diagnosis', 'target']).values
        y = data['target'].values
        return dataset_name,X,y
    else:
        raise Exception("No valid dataset name is inputted")

def wide_reach_classification(dataset_path, theta, epsilon_R=0.01, epsilon_P=0.01, epsilon_N=0.01):
    dataset_name, X, y = data_preprocessing(dataset_path=dataset_path)

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
    
    for i in range(num_samples):
        if y[i] == 1:  #P
            model.addConstr(x[i] <= 1 + sum(w[k] * X[i, k] for k in range(num_features)) - c - epsilon_P, name=f"classification_positive_{i}")
        else:  #N
            model.addConstr(y_vars[i] >= sum(w[k] * X[i, k] for k in range(num_features)) - c + epsilon_N, name=f"classification_negative_{i}")
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        nodes = model.NodeCount
        model.write(f'{dataset_name}.lp')
        clean_output(f'{dataset_name}.lp')

        print(f"{dataset_name} Dataset Results:")
        # print(f"Initial Reach: {initial_reach}")
        # print(f"BC Reach: {bc_reach}")
        print(f"Nodes: {nodes}\n")

        return {
            'Name': dataset_name,
            # 'Initial Reach': initial_reach,
            # 'BC Reach': bc_reach,
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
#results.append(wide_reach_classification(bc, theta=0.99))
results.append(wide_reach_classification(red_wine,theta=0.04))
results.append(wide_reach_classification(white_wine, theta=0.1))
#results.append(wide_reach_classification(crop, theta=0.9))

df_results = pd.DataFrame(results)
print("Summary of Results:")
print(df_results.to_string(index=False))