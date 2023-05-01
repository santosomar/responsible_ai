import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Load the MNIST dataset
mnist = fetch_openml("mnist_784")
X, y = mnist.data, mnist.target

# Scale the input data to the [0, 1] interval
X = X / 255.0

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Restricted Boltzmann Machine with 256 hidden units
rbm = BernoulliRBM(n_components=256, learning_rate=0.01, batch_size=10, n_iter=10, verbose=True, random_state=42)

# Initialize a logistic regression classifier
logistic = LogisticRegression(solver="newton-cg", multi_class="multinomial", random_state=42)

# Create a pipeline to first train the RBM and then train the logistic regression classifier
pipeline = Pipeline([("rbm", rbm), ("logistic", logistic)])

# Train the pipeline on the MNIST dataset
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the test set
accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
