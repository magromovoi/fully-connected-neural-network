import numpy as np
from scipy import special
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = np.maximum(input, 0)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return (input > 0) * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        self.output = special.expit(input)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * special.expit(input) * (1 - special.expit(input))


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.output = special.softmax(input, axis=1)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = special.softmax(input, axis=1)
        return softmax * (grad_output - (grad_output * softmax).sum(axis=1)[:, None])

class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.output = special.log_softmax(input, axis=1)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = special.softmax(input, axis=1)
        return softmax * (grad_output / softmax - grad_output.sum(axis=1)[:, None])

class LogSoftmax2(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        self.output = special.log_softmax(input, axis=1)
        return self.output

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = special.softmax(input, axis=1)
        eps = 1e-6
        return softmax * (grad_output / (softmax + eps) - grad_output.sum(axis=1)[:, None])


