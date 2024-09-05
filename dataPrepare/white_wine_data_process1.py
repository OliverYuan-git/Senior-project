import pandas as pd

file_path = 'winequality-white.csv'

data = pd.read_csv(file_path, delimiter=';')
print(data.head())
data.to_csv('winequality-white-corrected.csv', index=False)
