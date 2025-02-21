import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression function
def logistic_regression(X, y, iterations=200, lr=0.001):
    weights = np.zeros(X.shape[1])
    for _ in range(iterations):
        predictions = sigmoid(X @ weights)
        gradient = (X.T @ (predictions - y)) / y.size
        weights -= lr * gradient
    return weights

# Load dataset and preprocess
iris = load_iris()
X, y = iris.data[:, :2], (iris.target != 0).astype(int)  # Use first two features, binary classification

# Split and standardize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=9)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Train model
weights = logistic_regression(X_train, y_train)

# Predictions and accuracy
y_train_pred, y_test_pred = sigmoid(X_train @ weights) > 0.5, sigmoid(X_test @ weights) > 0.5
print(f'Training Accuracy: {np.mean(y_train_pred == y_train):.4f}')
print(f'Testing Accuracy: {np.mean(y_test_pred == y_test):.4f}')

# User input prediction
features = np.array(list(map(float, input("\nEnter sepal length and width: ").split()))).reshape(1, -1)
user_pred = sigmoid(scaler.transform(features) @ weights) > 0.5
print(f"Predicted species: {'Setosa' if user_pred == 0 else 'Versicolor/Virginica'}")

# Plot decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ weights) > 0.5
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', label='Train Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test Data')
plt.xlabel('Sepal length (standardized)')
plt.ylabel('Sepal width (standardized)')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.savefig('plot.png')
