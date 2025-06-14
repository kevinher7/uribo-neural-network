import numpy as np
from numpy.typing import NDArray
from typing import Self


class Layer:
    def __init__(self, num_neurons: int, *, output_layer: bool = False) -> None:
        self.num_neurons = num_neurons

        self.input_dim = 0
        self.output_layer = output_layer

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        # Append a bias unit to input data
        if self.augmentation:
            data = np.append(data, 1.0)

        activations = np.matmul(data, self.weights)
        self.outputs = self._activation_function(activations)

        return self.outputs

    def backward(
        self,
        deltas_next_layer: NDArray[np.float64],
        weights_to_next_layer: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # grad = np.matmul(self.outputs, self.weights)
        pass

    def build(self, input_dim, augmentation: bool) -> Self:
        """
        Creates connections with previous layer as a weight array
        """
        self.augmentation = augmentation

        if augmentation:
            input_dim += 1

        self.weights = np.random.randn(input_dim, self.num_neurons)

        return self

    def _activation_function(self, activations: float) -> NDArray[np.float64]:
        """Activation function. Currently only supports hyperbolic tangent"""
        return np.tanh(activations)  # (e^x - e^(-x))/(e^x + e^(-x))
