#! /home/abraham/anaconda3/envs/nlp_course/bin/python
#
#   Neural Network
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
import scipy.io as sio


# --------------------------------- sigmoid --------------------------------- #
#
#   DESCRIPTION:
#       Given an array with the values this function will perform
#       an element wise sigmoid operation
#
#   PARAMETERS:
#       values (IN) - A numpy array with the features
#
#   RETURNS:
#       A numpy array representing the sigmoid of each value
#
def sigmoid(values):
    return (1 / (1 + np.exp(-values)))


# ------------------------------ decode_nn_output --------------------------- #
#
#   DESCRIPTION:
#       Given a numpy array representing the output of a neural network, this
#       function will return a numpy array with the preicted values
#
#   PARAMETERS:
#       nn_output (IN) - A numpy array with the output of the neural netwokr
#
#   RETURNS:
#       A numpy array with the predicted values
#
#   NOTES:
#       nn_output is expected to be (m x c) where 'm' is the number of
#       predictions to be made and 'c' is the number of possible classes to be
#       predicted
#
#       The output of this function is (m x 1) where 'm' is the number of
#       predictions to be made. The values in this vector should be the
#       index of the max value for each row in nn_output.
#
#       Note that, we will start indexing from 1 to keep compatibily with
#       Octave
#
#       nn_output  = [0.3 -0.1 0.44]
#       prediction = [3]
#
#       Please note, due to the dataset, we will map 0 to 10
#
def decode_nn_output(nn_output):
    predictions = []
    # Iterate through the values
    for row in nn_output:
        prediction = np.argmax(row) + 1
        predictions.append(prediction)
    return np.transpose(np.array(predictions))


# -------------------------------- predict - -------------------------------- #
#
#   DESCRIPTION:
#       This function does the vectorized implementation of the feedforward
#       algorithm
#
#   PARAMETERS:
#       features       (IN) - A numpy array with the features
#       layer_weights  (IN) - A numpy array with the the weights of each layer
#
#   RETURNS:
#       A numpy array representing the predicted value for each set of
#       features
#
#       The dimentions of the features are (m x n) where 'm' is the number
#       of predictions to be done, and 'n' is the number of features
#
#       Each element of the weights array represents the weights of a
#       a layer. See the notes in the function
#
def predict(features, layer_weights):
    # Get the number of features
    num_features = len(features)

    # Define an array of inputs for each layer
    layer_inputs = [features]

    # Iterate through the neural network getting the outputs of each layer
    for layer_idx, layer_weight in enumerate(layer_weights):
        # Add the bias column to the layer_input
        layer_input = layer_inputs[layer_idx]
        ones = np.ones((num_features, 1))
        # We will transpose ones to use axis in np.insert
        ones = np.transpose(ones)
        layer_input = np.insert(layer_input, 0, ones, axis=1)

        # Define an array to store the output of this layer
        # Layer_output will be (m x N) where 'm' is the number of predictions
        # to be made and 'N' is the number of neurons
        layer_output = np.array([])
        # Get the vectorized output of the neuron
        for neuron_idx in range(len(layer_weight)):
            # Input layer is (m x (P+1)) where 'm' is the number of predictions
            # and 'P' is the number of neurons of the previous layer (P equals
            # number of features for the first layer)
            #
            # layer_weight for a given neuron is ((P+1) x 1) where 'P' is the
            # number of neurons in the previous layer (P equals number of
            # features for the first layer)
            #
            # neuron_output should therefore be (m x 1) where 'm' is the number
            # of predictions to made
            neuron_output = sigmoid(layer_input @ layer_weight[neuron_idx])
            # Add the output of this neuron into layer_output.
            # Each new neuron_output will be a new column in the layer_output
            # matrix
            #
            # layer_output is (m x N) where 'm' is the number of predictions
            # and 'N' is the number of neurons we have. Eventually layer_output
            # will become the layer_input
            if not layer_output.any():
                layer_output = neuron_output
            else:
                layer_output = np.column_stack((layer_output, neuron_output))

        # Have the output of this layer as input of the next one
        layer_inputs.append(layer_output)

    # If control reached this point we have passed through all the layers and
    # we should have an output of (m x c) where 'm' is the number of predictions
    # to be made and 'c' is the number of classes. Keep in mind that the last
    # layer of the NN should have 'c' neurons.
    neural_network_output = layer_output
    # We want the output of this function to be the predicted values in the
    # shape (m x 1) where 'm' is the number of predictions to be done;
    # therefore, we will decode the output
    return decode_nn_output(neural_network_output)

# --------------------------------- main ------------------------------------ #
# Read the dataset
# Each row in this dataset is a set of 400 pixels representing an image of a
# handwritten number in between 1 and 10.
dataset =  sio.loadmat('ex3data1.mat')
features = dataset['X']
# Each label is the actual numeric value.
# Note that for the number 10, this labels map to 0
labels = dataset['y']

# $$$TODO: Add back propagation algorithm, and comment the load of the weights
# Load the pre-trained weights
weights = sio.loadmat('ex3weights.mat')
layer_weights = [weights['Theta1'], weights['Theta2']]

predictions = predict(features, layer_weights)

# Show some predictions
while True:
    idx = input("Type the index of the label for which you want to check the "\
                "predicted value. q to exit \n")
    if idx in ['q', 'Q']:
        break

    assert idx.isdigit()
    idx = int(idx)

    print(f"Expected {labels[idx]}, predicted {predictions[idx]}")
