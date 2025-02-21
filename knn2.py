import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

class NaiveBayes:
    def fit(self, X, y):
        # Identify unique classes and calculate their mean, variance, and priors
        self.classes = np.unique(y)
        self.mean = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) for c in self.classes])
        self.priors = np.array([X[y == c].shape[0] / len(y) for c in self.classes])
    
    def predict(self, X):
        # Predict the class for each sample in X
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        # Calculate the posterior probability for each class
        posteriors = []
        for i in range(len(self.classes)):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        # Compute the Gaussian probability density function for each feature
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the Naive Bayes classifier
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Predict on the test set
y_pred = nb.predict(X_test)

# Calculate and display accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Map numerical predictions to species names
predicted_species = class_names[y_pred]
print("Predictions:", predicted_species)

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Get user input for a test example
print("\nEnter a test example for prediction:")
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
# For simplicity, set the remaining two features to 0 (model expects 4 features)
user_input = np.array([[sepal_length, sepal_width, 0, 0]])
user_pred = nb.predict(user_input)
predicted_species_user = class_names[user_pred][0]
print(f"Predicted species for the entered test example: {predicted_species_user}")
