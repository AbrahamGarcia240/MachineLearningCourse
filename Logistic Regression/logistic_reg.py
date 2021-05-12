#! /home/abraham/anaconda3/envs/nlp_course/bin/python
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------- normalize --------------------------------- #
#
def normalize(X):
    mu = X.mean()
    sigma = np.std(X)
    return (X - mu) / sigma, mu, sigma

# --------------------------------- sigmoid --------------------------------- #
#
def sigmoid(X, theta):
    h = X @ theta
    return (1 / (1 + np.exp(-h)))


# ---------------------------------- predict -------------------------------- #
#
def predict(X, theta):
    h = sigmoid(X, theta)
    new_h = []
    for elem in h:
        new_elem = 1 if elem >= 0.5 else 0
        new_h.append([new_elem])
    return np.array(new_h)

# ----------------------------------- cost ---------------------------------- #
#
def cost(h, y):
    m = len(h)
    return (1/m) * ((np.transpose(-y) @ np.log(h)) -\
                    (np.transpose(1-y) @ np.log(1-h)))


# ----------------------------- gradient_descend ---------------------------- #
#
def gradient_descend(X, y, theta, alpha):
    m = len(y)
    h = sigmoid(X, theta)
    gradient =  np.transpose(X) @ (h - y)
    theta =  theta - (alpha/m) * gradient
    return theta


# ================================= main ==================================== #
# Read the dataset
dataset = np.genfromtxt("ex2data2.txt", delimiter=",")

# Divide the dataset in X and y
X = dataset[:, 0:2]
y = dataset[:, 2:]

# Intialize theta with random values
theta = np.array([[random.random()], [random.random()]])

# Set a learning rate
alpha = 0.1

# Normalize the data
X, mu, sigma = normalize(X)
costs_array = []

# Train the model
for iter in range(400):
    print(f"Theta is {theta}")
    theta = gradient_descend(X, y, theta, alpha)
    h = sigmoid(X, theta)
    calculated_cost = cost(h, y)[0]
    costs_array.append(calculated_cost)
    print(f"Cost is: {calculated_cost}")
    print("")

# Do a prediction

X = np.array([0.051267, 0.69956])
X = (X - mu)/ sigma
h = predict(X, theta)
print(f"Expected 1 got {h}")

# Plot the error
plt.plot(costs_array)
plt.ylabel("Error of the model")
plt.show()

