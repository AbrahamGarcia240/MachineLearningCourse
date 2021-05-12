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

# ------------------------------- hypothesis -------------------------------- #
#
def hypothesis(X, theta):
    return X @ theta

# ----------------------------------- cost ---------------------------------- #
#
def cost(h, y):
    m = len(h)
    return (1/(2*m)) * sum((h - y)**2)


# ----------------------------- gradient_descend ---------------------------- #
#
def gradient_descend(X, y, theta, alpha):
    m = len(y)
    h = hypothesis(X, theta)
    gradient =  np.transpose(X) @ (h - y)
    theta =  theta - (alpha/m) * gradient
    return theta


# ================================= main ==================================== #
# Read the dataset
dataset = np.genfromtxt("ex1data2.txt", delimiter=",")

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
for iter in range(40):
    print(f"Theta is {theta}")
    theta = gradient_descend(X, y, theta, alpha)
    h = hypothesis(X, theta)
    calculated_cost = int(cost(h, y))
    costs_array.append(calculated_cost)
    print(f"Cost is: {calculated_cost}")
    print("")

# Do a prediction
X = np.array([1203, 3])
X = (X - mu)/ sigma
h = hypothesis(X, theta)
print(f"Expected close to 239500 got {h}")
print(f"Error of the prediction is {cost(h, np.array([239500]))}")

# Plot the error
plt.plot(costs_array)
plt.ylabel("Error of the model")
plt.show()

