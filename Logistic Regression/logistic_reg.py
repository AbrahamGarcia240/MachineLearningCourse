#! /home/abraham/anaconda3/envs/nlp_course/bin/python
#
#   Logistic Regression
#
#   This is a Python implementation of the Machine Learning algorithm
#   for classification.
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

# --------------------------------- sigmoid --------------------------------- #
#
#   DESCRIPTION:
#       Given an array with the values of the features and an array
#       with the weighted values of theta, this function returns the sigmoid
#       of features * theta
#
#   PARAMETERS:
#       features (IN) - A numpy array with the features
#       theta    (IN) - A numpy array with the theta values
#
#   RETURNS:
#       A numpy array representing the sigmoid of features * theta
#
#       The dimentions of the features are (m x n) where 'm' is the number
#       of predictions to be done, and 'n' is the number of features
#
#       The dimentions of theta are (n x 1) where 'n' is the number of features
#
def sigmoid(features, theta):
    hypothesis = features @ theta
    return (1 / (1 + np.exp(-hypothesis)))

# ---------------------------------- predict -------------------------------- #
#
#   DESCRIPTION:
#       Given an array of features and an array with the theta values, this
#       function will do a classification in between the class 0 or 1
#
#   PARAMETERS:
#       features (IN) - A numpy array with the features
#       theta    (IN) - A numpy array with the theta values
#
#   RETURNS:
#       A numpy array representing the prediction for each row of features
#
#   NOTES:
#       This function uses a threshold to use in order to classify the
#       features in between class 0 or 1. If the value of the sigmoid is
#       bigger of equal to the threshold it will be classified as 1
#
#       The dimentions of the features are (m x n) where 'm' is the number
#       of predictions to be done, and 'n' is the number of features
#
#       The dimentions of theta are (n x 1) where 'n' is the number of features
#
def predict(features, theta, threshold=0.5):
    hypothesis = sigmoid(features, theta)
    predictions = []
    for elem in hypothesis:
        new_elem = 1 if elem >= threshold else 0
        predictions.append([new_elem])
    return np.array(predictions)

# ----------------------------------- cost ---------------------------------- #
#
#   DESCRIPTION
#       Given a numpy array with the values of a Logistic Regression
#       hypothesis and the actual values to be predicted, this function
#       wil calculate the the error of the hypothesis
#
#   PARAMETERS:
#       h        (IN) - A numpy array with the values of the hypothesis, see
#                       notes
#       y        (IN) - A numpy array with the actual values to be predicted
#
#   RETURNS:
#       A numpy array representing the prediction for each row of features
#
#   NOTES:
#       The values is 'h' are result of calling the sigmoid function
#
#       The dimations of 'h' are (m x 1) where 'm' is the number of predictions
#       to be done
#
#       The dimentions of 'y' are (m x 1) where 'm' is the number of predictions
#       to be done
#
def cost(h, y):
    m = len(h)
    return (1/m) * ((np.transpose(-y) @ np.log(h)) -\
                    (np.transpose(1-y) @ np.log(1-h)))


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
def gradient_descent(features, labels, theta, alpha=0.03):
    m = len(labels)
    hypothesis = sigmoid(features, theta)
    # The gradient is the derivative of the cost function,
    # this is an (n x m) @ (m x 1) operation, which will give
    # an (n x 1) matrix, see notes.
    gradient =  np.transpose(features) @ (hypothesis - labels)
    # Use the gradient to update the values of theta
    theta =  theta - (alpha/m) * gradient
    return theta


# --------------------------------- main ------------------------------------ #
# Read the dataset
dataset = np.genfromtxt("ex2data2.txt", delimiter=",")

# Divide the dataset in features and labels
features = dataset[:, 0:2]
labels = dataset[:, 2:]

# Intialize theta with random values
theta = np.array([[random.random()], [random.random()]])

# Set a learning rate
alpha = 0.1

# Normalize the data
features, median, std_deviation = normalize(features)
costs_array = []

# Train the model
iterations = 50
for iter in range(iterations):
    # Print the actual value of theta before gradient descent
    print(f"Theta is {theta}")
    # Reduce the error
    theta = gradient_descent(features, labels, theta, alpha)
    # Calculate the hypothesis with the new values of theta
    hypothesis = sigmoid(features, theta)
    # See the cost (error) that we get with these new values of theta
    calculated_cost = cost(hypothesis, labels)[0]
    # Store the cost so that we can plot the cost function after we are done
    # with the iterations
    costs_array.append(calculated_cost)
    print(f"Cost is: {calculated_cost}")
    print("")

# If control reached this point we have trained the model for iterations times
# Now we will perform an actual prediction
test_features = np.array([0.051267, 0.69956])
# We need to normalize the features as we trained our model with normalized
# data
test_features = (test_features - median)/ std_deviation
# Do the actual prediction
prediction = predict(test_features, theta)
print(f"Expected 1 got {prediction}")

# Plot the evolution of the error through the training process to see if
# it got reduced, if not, try using a different learning rate
plt.plot(costs_array)
plt.ylabel("Error of the model")
plt.show()

