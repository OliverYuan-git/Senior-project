import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

X, y = generate_2_cluster_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

y_pred = lda.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Negative Samples',marker='+')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Positive Samples',marker='x')
plt.legend()
plt.title('2-Cluster Benchmark Data Distribution')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], c='blue', label='Negative Samples', marker='+')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c='red', label='Positive Samples', marker='x')
plt.scatter(X_test[y_pred != y_test][:, 0], X_test[y_pred != y_test][:, 1], facecolors='none', edgecolors='k', s=100, label='Misclassified')
plt.legend()
plt.title('LDA Classification Results on 2-Cluster Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
