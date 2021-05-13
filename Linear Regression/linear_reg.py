#! /home/abraham/anaconda3/envs/nlp_course/bin/python
#
#   Linear Regression
#
#   This is a Python implementation of the Machine Learning algorithm
#   for regression
#
#   The dataset was borrowed by the Machine Learning Coursera's course
#   by Andrew NG
#
#   This is meant to be a translation of the Octave implementation using
#   Numpy
#
#   Written by Abraham Garcia, 2021

# Imports
import numpy as np
import matplotlib.pyplot as plt
import random


# ------------------------------- normalize --------------------------------- #
#
#   DESCRIPTION;
#       Given a numpy array, this function will normalize the all its values
#       to be in between 0 and 1
#
#   PARAMETERS:
#       features (IN) - A numpy array with the features to be normalized
#
#   RETURNS:
#       The same dataset normalized, the mean and standard deviation of the
#       dataset
#
#   NOTES:
#       A given value x is normalized after performing
#
#           x_norm = (x - mean)/ (standard_deviation)
#
#       The dimentions of the features are (m x n) where 'm' is the number
#       of predictions to be done, and 'n' is the number of features
#
def normalize(features):
    median = features.mean()
    std_deviation = np.std(features)
    return (features - median) / std_deviation, median, std_deviation

# -------------------------------- predict - -------------------------------- #
#
#   DESCRIPTION:
#       This function does the vectorized implementation of hypothesis
#       caluculation
#
#   PARAMETERS:
#       features (IN) - A numpy array with the features
#       theta    (IN) - A numpy array with the theta values
#
#   RETURNS:
#       A numpy array representing the product of features * theta
#
#       The dimentions of the features are (m x n) where 'm' is the number
#       of predictions to be done, and 'n' is the number of features
#
#       The dimentions of theta are (n x 1) where 'n' is the number of features
#
def predict(features, theta):
    return features @ theta

# ----------------------------------- cost ---------------------------------- #
#
#   DESCRIPTION
#       Given a numpy array with the predictions and the actual values to be
#       predicted, this function wil calculate the the error of the hypothesis
#
#   PARAMETERS:
#       prediction (IN) - A numpy array with the values of the hypothesis, see
#                         notes
#       labels     (IN) - A numpy array with the actual values to be predicted
#
#       lambda_val (IN) - The value of lambda used to do regularization
#       theta      (IN) - A numpy array with the theta values
#
#
#   RETURNS:
#       A numpy array representing the prediction for each row of features
#
#   NOTES:
#       If lambda is not provided, no regularization will taken in consideration
#
#       The dimentions of predictions are (m x 1) where 'm' is the number of
#       predictions to be done
#
#       The dimentions of labels are (m x 1) where 'm' is the number of
#       predictions to be done
#
#       The dimentions of theta are (n x 1) where 'n' is the number of features
#
def cost(prediction, labels, lambda_val=0, theta=None):
    # Initialize the regularization value to be zero
    regularization = 0

    m = len(labels)
    # If a lambda value was provided we will calcuate the regularization
    if lambda_val and theta is not None:
        # To perform the operation we will transpose theta
        transposed_theta = np.transpose(theta)
        # Get rid of the theta values for the first features (x0)
        transposed_theta = transposed_theta[1:]
        # Calculate the regularization
        regularization = sum(transposed_theta ** 2)

    return (1 / (2 * m)) * (float(sum((predictions - labels) ** 2)) + \
                            regularization)

# ----------------------------- gradient_descent ---------------------------- #
#
#   DESCRIPTION:
#       This function does the algorithm of gradient_descent applying the
#       derivative of the cost function in order to minimize it
#
#   PARAMETERS:
#       features    (IN) - A numpy array with the features to be used to
#                          predict
#       labels      (IN) - A numpy array with the actual classes to be predicted
#                          i.e. the expected classes
#       theta       (IN) - A numpy array with the theta values of to be
#                          fitted
#       alpha       (IN) - An integer representing the learning rate
#       lambda_val  (IN) - The value of lambda used to do regularization
#
#   RETURNS:
#       The adapted theta values
#
#   NOTES;
#       The dimentions of the features are (m x n) where 'm' is the number
#       of predictions to be done, and 'n' is the number of features
#
#       The dimentions of labels are (m x 1) where 'm' is the number of
#       predictions to be done
#
#       The dimentions of theta are (n x 1) where 'n' is the number of features
#
# ----------------------------- gradient_descend ---------------------------- #
#
def gradient_descend(features, labels, theta, alpha=0.03, lambda_val=0):
    # Initialize the regularization value
    regularization = 1

    m = len(labels)
    hypothesis = predict(features, theta)
    # The gradient is the derivative of the cost function,
    # this is an (n x m) @ (m x 1) operation, which will give
    # an (n x 1) matrix, see notes.
    gradient =  np.transpose(features) @ (hypothesis - labels)
    # Check if we will do regularization
    if lambda_val:
        regularization = (1 - (alpha * lambda_val) / m)
    # Use the gradient to update the values of theta
    theta =  (theta * regularization) - (alpha/m) * gradient
    return theta


# --------------------------------- main ------------------------------------ #
# Read the dataset
dataset = np.genfromtxt("ex1data2.txt", delimiter=",")

# Divide the dataset in features and labels
features = dataset[:, 0:2]
labels = dataset[:, 2:]

# Intialize theta with random values
theta = np.array([[random.random()], [random.random()]])

# Set a learning rate
alpha = 0.1

# Define a value of lambda (to do regularization and avoid overfitting)
lambda_val = 0.1

# Normalize the data
features, median, std_deviation = normalize(features)
costs_array = []

# Train the model
iterations = 50
for iter in range(iterations):
    # Print the actual value of theta before gradient descent
    print(f"Theta is {theta}")
    # Reduce the error
    theta = gradient_descend(features, labels, theta, alpha, lambda_val)
    # Calculate the hypothesis with the new values of theta
    predictions = predict(features, theta)
    # See the cost (error) that we get with these new values of theta
    calculated_cost = int(cost(predictions, labels, lambda_val, theta))
    # Store the cost so that we can plot the cost function after we are done
    # with the iterations
    costs_array.append(calculated_cost)
    print(f"Cost is: {calculated_cost}")
    print("")

# If control reached this point we have trained the model for iterations times
# Now we will perform an actual prediction
test_features = np.array([1203, 3])
# We need to normalize the features as we trained our model with normalized
# data
test_features = (test_features - median) / std_deviation
# Do the actual prediction
prediction = predict(test_features, theta)
print(f"Expected close to 239500 got {prediction}")
print(f"Error of the prediction is {cost(prediction, np.array([239500]))}")


# Plot the evolution of the error through the training process to see if
# it got reduced, if not, try using a different learning rate
plt.plot(costs_array)
plt.ylabel("Error of the model")
plt.show()

