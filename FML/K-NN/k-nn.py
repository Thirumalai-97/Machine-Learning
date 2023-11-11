import numpy as np 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mnist =fetch_openml('mnist_784')

X = mnist.data
y = mnist.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with k=5
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


random_index=np.random.randint(0,len(X_test))
print(random_index)
print(X_test.shape)

X_test=np.array(X_test)

image =X_test[random_index,:].reshape(28,28)
label=y_pred[random_index]

plt.figure(figsize=(5,5))
plt.imshow(image, cmap='gray')
plt.title(f"Label : {label}")
plt.axis('off')
plt.show()


