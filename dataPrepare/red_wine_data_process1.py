import pandas as pd

file_path = 'winequality-red.csv'

data = pd.read_csv(file_path, delimiter=';')
print(data.head())
data.to_csv('winequality-red-corrected.csv', index=False)
