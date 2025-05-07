import numpy as np
from numpy.typing import NDArray

from src.model.layer import Layer


class NeuralNetwork:
    def __init__(self, input_dim: int, layers: list[Layer], *, augmentation: bool = True) -> None:
        """Initialize NN by assigning input dimensions to all Layers"""
        if augmentation:
            input_dim += 1
            # Add one neuron to each layer except the output layer
            for layer in layers[:-1]:
                layer.num_neurons += 1

        # Remove the first layer (hidden) and append it to NN
        first_hidden_layer = layers.pop(0)
        first_hidden_layer.input_dim = input_dim
        first_hidden_layer.build()

        self.layers = [first_hidden_layer]

        for index, layer in enumerate(layers):
            if index == len(layers) - 1:
                layer.output_layer = True

            # Give input dimension as the previous layer's number of neurons
            layer.input_dim = self.layers[index].num_neurons
            layer.build()

            self.layers.append(layer)

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
        # Add bias neuron to input data
        data = np.append(data, [1])

        layer_output = data
        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        if classification:
            output = self._softmax(layer_output)
        else:
            output = layer_output

        return output

    def _softmax() -> None:
        raise NotImplementedError(
            "Neural Network does not support classification (only regression)"
        )
