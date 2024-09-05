import pandas as pd

file_path = r'C:\Users\OLIVER YUAN\Desktop\reseach\winequality-white.csv'

data = pd.read_csv(file_path, delimiter=';')
print(data.head())
data.to_csv(r'C:\Users\OLIVER YUAN\Desktop\reseach\winequality-white-corrected.csv', index=False)
