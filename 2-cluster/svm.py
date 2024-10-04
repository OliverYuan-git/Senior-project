import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to generate 2-cluster data
def generate_2_cluster_data(n=400, d=8, d_max=8):
    negative_samples = np.random.rand(4 * n // 5, d)
    s = (np.math.factorial(d) / np.math.factorial(d_max)) ** (1 / d)  
    positive_uniform = np.random.rand(n // 10, d)
    positive_cluster_1 = np.random.rand(n // 20, d)
    positive_cluster_1 = positive_cluster_1 / positive_cluster_1.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = np.random.rand(n // 20, d)
    positive_cluster_2 = positive_cluster_2 / positive_cluster_2.sum(axis=1, keepdims=True) * s
    positive_cluster_2 = s - positive_cluster_2
    positive_samples = np.vstack((positive_uniform, positive_cluster_1, positive_cluster_2))
    X = np.vstack((positive_samples, negative_samples))
    y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))
    return X, y

# Generate data
X, y = generate_2_cluster_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train RBF SVM using all 8 dimensions
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale') 
svm_rbf.fit(X_train, y_train)

# Predict and evaluate using all 8 dimensions
y_pred = svm_rbf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Set up mesh grid for plotting decision boundary (only for the first two dimensions)
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# We can still calculate the decision function on the 8-dimensional data, but only plot 2D
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 6))])  # Set the rest of the dimensions to 0
Z = Z.reshape(xx.shape)

# Plot the decision boundary and margin
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange', alpha=0.5)

# Plot the original data points (only first two dimensions)
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', label='Negative Samples', marker='+')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='red', label='Positive Samples', marker='x')

# Highlight the misclassified points (first two dimensions)
plt.scatter(X_test[y_pred != y_test][:, 0], X_test[y_pred != y_test][:, 1], facecolors='none', edgecolors='k', s=100, label='Misclassified')

# Highlight the support vectors (only first two dimensions)
plt.scatter(svm_rbf.support_vectors_[:, 0], svm_rbf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='yellow', label='Support Vectors')

# Add titles and labels
plt.legend()
plt.title('SVM Classification Results with RBF Kernel and Decision Boundary (First Two Dimensions)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()