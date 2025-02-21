import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# KNN prediction function
def knn_predict(X_train, y_train, test_point, k=3):
    distances = np.linalg.norm(X_train - test_point, axis=1)
    nearest_neighbors = np.argsort(distances)[:k]
    return np.bincount(y_train[nearest_neighbors]).argmax()

# Predict and calculate accuracy
predictions = np.array([knn_predict(X_train, y_train, x) for x in X_test])
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

# User input
features = list(map(float, input("\nEnter 4 features (space-separated): ").split()))
predicted_class = knn_predict(X_train, y_train, np.array(features))
print(f"Predicted class: {iris.target_names[predicted_class]}")

