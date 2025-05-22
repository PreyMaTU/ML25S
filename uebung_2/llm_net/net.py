
# This code was generated using ChatGPT based on the following prompt:
# Please write a neural net in python from scratch in Python (you may use math 
# libs like numpy, but no ML framework like sklearn or torch) with the following 
# configuration:
# fully connected feed forward; configurable number of layers; configurable 
# number of perceptrons per layer; MSE loss; ReLu activation on all layers except 
# the last, on the last use Sigmoid; Gradient Decent optimization; train using 
# minibatches of size 32; configurable learning rate; configurable number of 
# epochs;
# The net is going to be used for classification and needs to output the label as 
# perceptrons with probabilities per label. The inputs are x_train, x_test, 
# y_train and y_test, which are numpy 2D arrays where each sample is a row and 
# each column is a feature. The net should size itself on creation based on the 
# number of input samples and number of unique labels in the y_train data.
# You do not have to design your code to be a reusable framework, it may be 
# tightly integrated.

# 

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

class NeuralNet:
    def __init__(self, x_train, y_train, layer_sizes, learning_rate=0.01):
        self.input_size = x_train.shape[1]
        self.output_size = len(np.unique(y_train))
        self.learning_rate = learning_rate

        # Add input and output layers
        self.layer_sizes = [self.input_size] + layer_sizes + [self.output_size]
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2. / self.layer_sizes[i])
            bias = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        activations = [x]
        pre_activations = []

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            if i == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = relu(z)
            activations.append(a)

        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        delta = mse_loss_deriv(y_true, activations[-1]) * sigmoid_deriv(pre_activations[-1])
        for i in reversed(range(len(self.weights))):
            grads_w[i] = np.dot(activations[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_deriv(pre_activations[i - 1])

        return grads_w, grads_b

    def update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, x_train, y_train, epochs=100):
        batch_size = 32
        num_classes = self.output_size
        y_train_encoded = y_train # one_hot_encode(y_train, num_classes)

        for epoch in range(epochs):
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train_encoded[indices]

            for i in range(0, len(x_train), batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                activations, pre_activations = self.forward(x_batch)
                grads_w, grads_b = self.backward(activations, pre_activations, y_batch)
                self.update_weights(grads_w, grads_b)

            # Optionally print loss
            train_pred = self.predict(x_train)
            loss = mse_loss(y_train_encoded, train_pred)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def evaluate(self, x_test, y_test):
        y_pred_probs = self.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test = np.argmax(y_test, axis=1)
        accuracy = np.mean(y_pred == y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

# Example usage:
# Define parameters
# layer_sizes = [64, 32] # Hidden layers
# learning_rate = 0.01
# epochs = 50

# nn = NeuralNet(x_train, y_train, layer_sizes=layer_sizes, learning_rate=learning_rate)
# nn.train(x_train, y_train, epochs=epochs)
# nn.evaluate(x_test, y_test)
