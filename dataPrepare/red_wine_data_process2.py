import pandas as pd

file_path = 'winequality-red-corrected.csv'
data = pd.read_csv(file_path)

num_rows = data.shape[0]
num_features = data.shape[1] - 1 

print(f"col）：{num_rows}")
print(f"features：{num_features}")
print(data.info())

# check missing value
missing_values = data.isnull().sum()
print("missing list: ")
print(missing_values)

#standard dev graph
import matplotlib.pyplot as plt

std_deviation = data.drop(columns=['quality']).std()

plt.figure(figsize=(10, 6))
std_deviation.plot(kind='bar')
plt.title('Standard Deviation of Each Feature')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=45)
plt.show()

import seaborn as sns

# correlation graph
correlation_matrix = data.drop(columns=['quality']).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

