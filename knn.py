import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# KNN function
def knn_predict(X_train, y_train, test_point, k=3):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(test_point, X_train[i])
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])  # Sort by distance
    k_nearest = distances[:k]  # Take k nearest neighbors
    classes = [neighbor[1] for neighbor in k_nearest]  # Get their labels
    return max(set(classes), key=classes.count)  # Return the most common class

# Predict on the test set
predictions = [knn_predict(X_train, y_train, test_point, k=3) for test_point in X_test]

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

# User input for prediction
print("\nEnter features for a test example:")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))
test_point = np.array([sepal_length, sepal_width, petal_length, petal_width])

# Predict class for user input
predicted_class = knn_predict(X_train, y_train, test_point, k=3)
print(f"Predicted class: {iris.target_names[predicted_class]}")
