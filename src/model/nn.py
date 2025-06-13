import numpy as np
from numpy.typing import NDArray

from src.model.layer import Layer


class NeuralNetwork:
    def __init__(self, input_dim: int, layers: list[Layer], *, augmentation: bool = True) -> None:
        """Initialize NN by assigning input dimensions and building all Layers"""
        self.augmentation = augmentation

        self.layers: list[Layer] = []
        for layer in layers[:-1]:
            layer.build(input_dim, augmentation)
            self.layers.append(layer)
            input_dim = layer.num_neurons

        # Build output layer without augmentation
        layers[-1].build(input_dim, False)
        self.layers.append(layers[-1])

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
        layer_output = data
        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        if classification:
            self.output = self._softmax(layer_output)
        else:
            self.output = layer_output

        return self.output

    def _backward(self, target_data: NDArray[np.float64]) -> None:
        """
        Backward Propagation to calculate gradients
        """
        # Calculate deltas for output layer
        delta_next_layer = self.output - target_data

        # Backpropagate through the remaining hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            delta_next_layer = self.layers[i].backward(delta_next_layer, self.layers[i + 1].weights)

    def _softmax() -> None:
        raise NotImplementedError(
            "Uribo Neural Network does not support classification (only regression)"
        )
