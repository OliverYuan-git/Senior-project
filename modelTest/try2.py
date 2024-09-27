import random
import pandas as pd
from gurobipy import Model, GRB, quicksum, LinExpr
from sklearn.model_selection import train_test_split
import numpy as np

#crop_data = pd.read_csv('WinnipegDataset.csv')
#bc_data = pd.read_csv('wdbc.csv')

# data processing
red_wine_data['target'] = (red_wine_data['quality'] >= 8).astype(int)
X_red = red_wine_data.drop(columns=['quality', 'target']).values
y_red = red_wine_data['target'].values

white_wine_data['target'] = (white_wine_data['quality'] >= 8).astype(int)
X_white = white_wine_data.drop(columns=['quality', 'target']).values
y_white = white_wine_data['target'].values

#bc_data['target'] = (bc_data['diagnosis'] == 'M').astype(int)
#X_bc = bc_data.drop(columns=['diagnosis', 'target']).values
#y_bc = bc_data['target'].values

crop_random_sample_list = random.sample(range(325835), 10000)
#crop_data = crop_data.iloc[crop_random_sample_list]
#crop_data['target'] = (crop_data['label'] == 6).astype(int)
#X_crop = crop_data.drop(columns=['label', 'target']).values
#y_crop = crop_data['target'].values

# randon sample
X_red_sample, _, y_red_sample, _ = train_test_split(X_red, y_red, test_size=0.5, random_state=50)
X_white_sample, _, y_white_sample, _ = train_test_split(X_white, y_white, test_size=0.5, random_state=50)
#X_crop_sample, _, y_crop_sample, _ = train_test_split(X_crop, y_crop, test_size=0.5, random_state=42)
#X_bc_sample, _, y_bc_sample, _ = train_test_split(X_bc, y_bc, test_size=0.5, random_state=42)


# Adjust Lambda
def compute_lambda(n, theta):
    return 10 * (n + 1) * theta

# initial num for Gurobi 
def generate_initial_solution(num_samples):
    return [0.5 for _ in range(num_samples)]

def wide_reach_classification(X, y, dataset_name, theta, epsilon_R=0.01, epsilon_P=0.01, epsilon_N=0.01):
    # get lambda
    lambda_value = compute_lambda(len(y), theta)

    model = Model("Wide-Reach_Classification")
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

    for v in model.getVars():
        print('%s %g' % (v.VarName, v.X))

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        initial_reach = sum(1 for i in range(num_samples) if y[i] == 1)
        # bc_reach = sum(x[i].X for i in range(num_samples) if x[i].X > 0.5)
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
        with open('cleaned_output.lp', 'w') as file:
            file.write(cleaned_content)
        print("Cleaned LP file created as 'cleaned_output.lp'") 
        print(f"{dataset_name} Dataset Results:")
        print(f"Initial Reach: {initial_reach}")
        # print(f"BC Reach: {bc_reach}")
        print(f"Nodes: {nodes}\n")

        return {
            'Name': dataset_name,
            'Initial Reach': initial_reach,
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


    # if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    #     try:
    #         x_values = model.getAttr('X', x)
    #         with open('output.lp', 'w') as f:
    #             f.write(f"\\ Model Wide-Reach_Classification\n")
    #             f.write(f"\\ LP format - for model browsing.\n")
    #             f.write(f"Maximize\n")
    #             # Write only non-zero x
    #             non_zero_vars = [f"x[{i}]" for i in range(num_samples) if x_values[i] > 1e-6]
    #             f.write(" + ".join(non_zero_vars))
    #             f.write("\n")
            
    #         initial_reach = sum(1 for i in range(num_samples) if y[i] == 1)
    #         nodes = model.NodeCount
    #         print(f"{dataset_name} Dataset Results:")
    #         print(f"Initial Reach: {initial_reach}")
    #         print(f"Nodes: {nodes}\n")

    #         return {
    #             'Name': dataset_name,
    #             'Initial Reach': initial_reach,
    #             'Nodes': nodes
    #         }
    #     except Exception as e:
    #         print(f"Error while retrieving variable values: {e}")
    #         return {
    #             'Name': dataset_name,
    #             'Initial Reach': 0,
    #             'Nodes': 0
    #         }
    # else:
    #     print(f"No feasible solution found for {dataset_name}. Status code: {model.status}")
    #     return {
    #         'Name': dataset_name,
    #         'Initial Reach': 0,
    #         'Nodes': 0
    #     }

results = []
#results.append(wide_reach_classification(X_bc_sample, y_bc_sample, "B&C", theta=0.99))
results.append(wide_reach_classification(X_red, y_red, "Wine Quality (red)", theta=0.04))
results.append(wide_reach_classification(X_white, y_white, "Wine Quality (white)", theta=0.1))
#results.append(wide_reach_classification(X_crop_sample, y_crop_sample, "Crop", theta=0.9))

df_results = pd.DataFrame(results)
print("Summary of Results:")
print(df_results.to_string(index=False))
