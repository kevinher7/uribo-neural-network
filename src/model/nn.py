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

    def train(
        self, data: NDArray[np.float64], targets: NDArray[np.float64], *, epochs: int
    ) -> NDArray[np.float64]:
        for _e in range(epochs):
            output = self.forward(data)
            r = self._backward(data, targets, output)

        return 2

    def forward(self, data: NDArray[np.float64], *, classification: bool = False) -> float:
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
        if self.augmentation:
            data = np.append(data, 1)

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

        input_grad = np.outer(deltas_next_layer, input_data.T)

    def _softmax() -> None:
        raise NotImplementedError(
            "Uribo Neural Network does not support classification (only regression)"
        )
