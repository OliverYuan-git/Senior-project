import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = 'cgan_white_wine_7.csv'
data = pd.read_csv(file_path)

# Split the dataset into features and labels
X = data.drop('quality', axis=1)  # Features
y = data['quality']  # Labels

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set for Linear Regression
y_pred_lr = lr_model.predict(X_test)

# Output the performance metrics for Linear Regression
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print(f"Linear Regression Mean Squared Error: {mse:.2f}")
print(f"Linear Regression R^2 Score: {r2:.2f}")

# Plotting residuals (Actual - Predicted) to visualize the errors
residuals = y_test - y_pred_lr
plt.figure(figsize=(10, 6))

plt.scatter(y_pred_lr, residuals, alpha=0.5, color='purple', edgecolor='k')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel("Predicted Quality", fontsize=14)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=14)
plt.title("Residual Plot (Linear Regression)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
