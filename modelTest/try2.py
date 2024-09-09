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
