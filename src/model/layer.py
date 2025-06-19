import numpy as np
from numpy.typing import NDArray
from typing import Self


class Layer:
    def __init__(self, num_neurons: int, *, output_layer: bool = False) -> None:
        self.num_neurons = num_neurons
        self.is_output = False
        self.grad = np.zeros(1)

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.inputs = data
        activations = np.matmul(data, self.weights)

        if self.is_output:
            # Skip activation and return
            self.outputs = activations
            return self.outputs

        self.outputs = self._activation_function(activations)

        if self.augmentation:
            # Append bias unit to outputs
            self.outputs = np.append(self.outputs, 1.0)

        return self.outputs

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
