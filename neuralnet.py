import numpy as np
import matplotlib.pyplot as plt

class Neuralnet:
	def __init__(self, neurons):
		self.layers = len(neurons)

		# Learning rate
		self.rate = .05

		# Input vectors
		self.inputs = []
		# Output vectors
		self.outputs = []
		# Error vectors
		self.errors = []
		# Weight matrices
		self.weights = []
		# Bias vectors
		self.biases = []

		for layer in range(self.layers):
			# Create the input, output, and error vector
			self.inputs.append(np.empty(neurons[layer]))
			self.outputs.append(np.empty(neurons[layer]))
			self.errors.append(np.empty(neurons[layer]))

		for layer in range(self.layers - 1):
			# Create the weight matrix
			self.weights.append(np.random.normal(
				scale=1.0/np.sqrt(neurons[layer]),
				size=[neurons[layer], neurons[layer + 1]]
			))
			# Create the bias vector
			self.biases.append(np.random.normal(
				scale=1.0/np.sqrt(neurons[layer]),
				size=neurons[layer + 1]
			))

	def feedforward(self, inputs):
		# Set input neuron inputs
		self.inputs[0] = inputs
		for layer in range(self.layers - 1):
			# Find output of this layer from its input
			self.outputs[layer] = np.tanh(self.inputs[layer])
			# Find input of next layer from output of this layer and weight matrix (plus bias)
			self.inputs[layer + 1] = np.dot(self.weights[layer].T, self.outputs[layer]) + self.biases[layer]
		self.outputs[-1] = np.tanh(self.inputs[-1])

	def backpropagate(self, targets):
		# Calculate error at output layer
		self.errors[-1] = self.outputs[-1] - targets
		# Calculate error vector for each layer
		for layer in reversed(range(self.layers - 1)):
			gradient = 1 - self.outputs[layer] * self.outputs[layer]
			self.errors[layer] = gradient * np.dot(self.weights[layer], self.errors[layer + 1])
		# Adjust weight matrices and bias vectors
		for layer in range(self.layers - 1):
			self.weights[layer] -= self.rate * np.outer(self.outputs[layer], self.errors[layer + 1])
			self.biases[layer] -= self.rate * self.errors[layer + 1]

# Create a neural network that accepts 262 bits as input and has 1 output neuron
net = Neuralnet([262, 150, 1])

positions = []
scores = []

# Train neural network on entire data set multiple times
for epoch in range(1):
	# Total error for this epoch
	error = 0
	for index in range(len(positions)):
		# Extract input data
		game = positions[index]
		# Feed input data to neural network
		net.feedforward(game)
		# Target output is the position score
		score = scores[index]
		# Train neural network based on target output
		net.backpropagate(score)
		error += np.sum(net.errors[-1] * net.errors[-1])
	print 'Epoch ' + str(epoch) + ' error: ' + str(error)

# print "Input neurons:" + str(net.inputs)







