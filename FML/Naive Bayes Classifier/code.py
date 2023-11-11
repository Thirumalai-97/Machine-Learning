import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load and preprocess your data
data = pd.read_csv('data.csv')

# Split data into features (X) and the class labels (y)
X = data.drop(columns=['Class Label'])  # features(X)
y = data['Class Label']  # Class labels

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Initialize and train the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Validate the model on the validation set
validation_accuracy = nb_classifier.score(X_val, y_val)

# Test the model on the test set
test_accuracy = nb_classifier.score(X_test, y_test)

print(f"Validation Accuracy: {validation_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Calculate the mean and covariance matrix for each class
unique_classes = np.unique(y_train)
class_means = {}
covariance_matrices = {}

for i in unique_classes:
    mask = (y_train == i)
    class_data = X_train[mask]
    
    # Calculate mean for each feature in the class
    class_mean = np.mean(class_data, axis=0)
    class_means[i] = class_mean
    
    # Calculate covariance matrix
    covariance_matrix = np.cov(class_data, rowvar=False)
    covariance_matrices[i] = covariance_matrix

print("Class Means:")
for i, class_mean in class_means.items():
    print(f"Class {i}:\n{class_mean}")

print("\nCovariance Matrices:")
for i, covariance_matrix in covariance_matrices.items():
    print(f"Class {i}:\n{covariance_matrix}")
