import numpy as np
from .base import Criterion
from .activations import LogSoftmax, LogSoftmax2
from scipy import special


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, f"input and target shapes not matching {input.shape}, {target.shape}"
        self.output = ((input - target) ** 2).mean()
        return self.output

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, f"input and target shapes not matching {input.shape}, {target.shape}"
        batch_size, n = input.shape
        return 2 * (input - target) / (batch_size * n)


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size, num_classes = input.shape
        return -(self.log_softmax.compute_output(input) \
                * (np.tile(target[:, None], num_classes) == range(num_classes))).sum() / batch_size
        

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size, num_classes = input.shape
        grad_output = (np.tile(target[:, None], num_classes) == range(num_classes))
        return self.log_softmax.compute_grad_input(input, grad_output) / batch_size * (-1)


class CrossEntropyLoss2(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax2()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size, num_classes = input.shape
        return -(self.log_softmax.compute_output(input) \
                * (np.tile(target[:, None], num_classes) == range(num_classes))).sum() / batch_size
        

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size, num_classes = input.shape
        grad_output = (np.tile(target[:, None], num_classes) == range(num_classes))
        return self.log_softmax.compute_grad_input(input, grad_output) / batch_size * (-1)






