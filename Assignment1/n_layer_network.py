from three_layer_neural_network import NeuralNetwork, ACTIVATIONS, dACTIVATIONS, generate_data
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-11


def generate_data2():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.1)
    return X, y


class DenseLayer(object):

    def __init__(self, input_size, output_size, activation='linear', reg_lambda=0.01, seed=0):
        self.input_size = input_size
        self.output_size = output_size
        np.random.seed(seed)
        self.W = np.random.randn(
            self.input_size, self.output_size) / np.sqrt(self.input_size)
        self.b = np.zeros((1, self.output_size))

        self.actFun_type = activation
        self.reg_lambda = reg_lambda

        self.delta = None
        self.x = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None

    def actFun(self, z):
        '''
        actFun computes the activation functions
        :param z: net input
        :return: activations
        '''

        return ACTIVATIONS[self.actFun_type](z)

    def diff_actFun(self, z):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :return: the derivatives of the activation functions wrt the net input
        '''

        return dACTIVATIONS[self.actFun_type](z)

    def feedforward(self, x):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        self.x = x
        self.z = np.dot(self.x, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a

    def __call__(self, x):
        return self.feedforward(x)

    def backprop(self, delta_piece):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param delta_piece: delta from previous layer without activation gradient
        :return: The delta without multiplication by activation gradient
        '''

        da = self.diff_actFun(self.z)
        delta = delta_piece * da
        self.dW = np.dot(self.x.T, delta) + self.reg_lambda * self.W
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)

    def update(self, epsilon):
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db


class SoftmaxLayer(DenseLayer):

    def __init__(self, input_size, output_size, reg_lambda=0.01, seed=0):
        super(SoftmaxLayer, self).__init__(input_size, output_size, 'linear',
                                           reg_lambda, seed)
        self.probs = None

    def actFun(self, z):
        '''
        actFun computes the sofmtax activation functions
        :param z: net input
        :return: activations
        '''
        exp_scores = np.exp(z - np.max(z))
        return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True))

    def diff_actFun(self, z):
        '''
        diff_actFun computes the derivatives of the softmax functions wrt the layer input
        :param z: net input
        :return: the derivatives of the softmax functions wrt the net input
        '''

        raise ValueError("Softmax is only meant to be used as final layer")

    def backprop(self, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param x: input data
        :param y: given labels - Shape (samples,)
        '''

        num_examples = len(self.x)
        delta = self.a
        delta[range(num_examples), y] -= 1

        dregW = (self.reg_lambda * self.W)
        self.dW = np.dot(self.x.T, delta) + dregW
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)


class DeepNeuralNetwork(NeuralNetwork):

    def __init__(self, layers, seed=0):
        '''
        :param layers: the layers in sequential order of the neural network
        :param seed: random seed
        '''
        self.layers = layers
        self.input_dim = self.layers[0].input_size
        self.output_dim = self.layers[-1].output_size
        self.probs = None

    def feedforward(self, X):
        assert self.input_dim == X.shape[1], "Net input size is %s but passed array with %s features" % (
            self.input_dim, X.shape[1])
        tensor = X
        for layer in self.layers:
            tensor = layer(tensor)
        self.probs = tensor
        return tensor

    def __call__(self, X):
        return self.feedforward(X)

    def backprop(self, y):
        delta_piece = y
        for layer in reversed(self.layers):
            delta_piece = layer.backprop(delta_piece)

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels - Shape (samples,)
        :return: the loss for prediction
        '''
        num_examples = len(X)
        # Forward propagation
        self(X)
        data_loss = np.sum(-np.log(self.probs[np.arange(num_examples), y]))
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self(X)
            # Backpropagation
            self.backprop(y)

            # Gradient descent parameter update
            for layer in self.layers:
                layer.update(epsilon)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" %
                      (i, self.calculate_loss(X, y)))

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self(X)
        return np.argmax(self.probs, axis=1)


def main():
    # # generate and visualize Make-Moons dataset
    # X, y = generate_data()
    X, y = generate_data2()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
    sizes = [X.shape[1], 10, 6, 4, 3, 3]
    layers = [DenseLayer(sizes[i], sizes[i + 1], activation='ramp')
              for i in range(len(sizes) - 1)]
    layers.append(SoftmaxLayer(sizes[-1], 2))
    model = DeepNeuralNetwork(layers)

    model.fit_model(X, y, epsilon=0.001, num_passes=50000)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
