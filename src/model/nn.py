import numpy as np
from numpy.typing import NDArray

from src.model.layer import Layer


class NeuralNetwork:
    def __init__(self, input_dim: int, layers: list[Layer], *, augmentation: bool = True) -> None:
        """Initialize NN by assigning input dimensions and building all Layers"""
        self.augmentation = augmentation

        self.layers: list[Layer] = []
        for layer in layers:
            layer.build(input_dim, augmentation)
            self.layers.append(layer)
            input_dim = layer.num_neurons

        # Set last layer as output layer
        self.layers[-1].is_output = True

    def train(
        self, data: NDArray[np.float64], targets: NDArray[np.float64], *, epochs: int
    ) -> NDArray[np.float64]:
        if self.augmentation:
            data = np.append(data, 1)

        for _e in range(epochs):
            output = self._forward(data)
            print(f"Model output: {output}")
            self._backward(data, targets, output)

        return

    def _forward(self, data: NDArray[np.float64], *, classification: bool = False) -> float:
        """
        Forward method for layer class.
        Args:
            data (NDArray[np.float64]): 1 dimmensional numpy array

        Kwargs:
            classification (bool): determines whether to use regression or classification

        Returns:
            output (float): for regression
        """
        # Understood as the "input layer" output
        layer_output = data

        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        if self.augmentation:
            layer_output = layer_output[:-1]

        if classification:
            return self._softmax(layer_output)
        else:
            return layer_output

    def _backward(
        self,
        input_data: NDArray[np.float64],
        target_data: NDArray[np.float64],
        model_output: NDArray[np.float64],
    ) -> None:
        """
        Backward Propagation to calculate gradients
        """
        # Calculate deltas for output layer
        deltas_next_layer = model_output - target_data

        # Backpropagate through the remaining hidden layers
        for i in range(len(self.layers) - 1):
            deltas_next_layer = self.layers[-(i + 2)].backward(
                deltas_next_layer, self.layers[-(i + 1)].weights
            )

        # Calculate gradient at input layer level
        input_grad = np.outer(deltas_next_layer, input_data.T)
        # print([layer.grad for layer in self.layers])
        # print([layer.grad.shape for layer in self.layers])
        self._update_weights(input_grad)

    def _softmax() -> None:
        raise NotImplementedError(
            "Uribo Neural Network does not support classification (only regression)"
        )

    # TODO: Consider updating weights (and store them in dummy variables)
    # while backpropagation continues to improve time complexity
    def _update_weights(
        self, input_grad: NDArray[np.float64], learning_rate: float = 0.1, optimizer: str = "SGD"
    ) -> None:
        """
        Updates weights via the provided optimizer.
        Currently only supports Stochastic Gradient Descent
        """
        # Update input layer weights
        self.layers[0].weights -= learning_rate * input_grad.T
        for layer_num in range(len(self.layers[1:])):
            self.layers[-layer_num - 1].weights -= (
                learning_rate * self.layers[-layer_num - 2].grad.T
            )
