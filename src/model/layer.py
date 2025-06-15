import numpy as np
from numpy.typing import NDArray
from typing import Self


class Layer:
    def __init__(self, num_neurons: int, *, output_layer: bool = False) -> None:
        self.num_neurons = num_neurons
        self.grad = np.zeros(1)

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        activations = np.matmul(data, self.weights)
        self.outputs = self._activation_function(activations)

        if self.augmentation:
            # Append bias unit to outputs
            self.outputs = np.append(self.outputs, 1.0)

        return self.outputs

    def backward(
        self,
        deltas_next_layer: NDArray[np.float64],
        weights_to_next_layer: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Calculate the gradient
        # grad_j = d_k*z_j
        self.grad = np.outer(deltas_next_layer, self.outputs.T)
        # grad E_kj
        # Row: k (next layer unit)
        # Column: j (current layer unit)

        # Calculate this layer's delta
        if self.augmentation:
            # Remove the bias related parameters (bias delta is not needed)
            # since no units send connections to it
            # (so this delta is not involved gradient calculations)
            self.outputs = self.outputs[:-1]
            weights_to_next_layer = weights_to_next_layer[:-1]

        # Since we use hyperbolic tangent as the activation function
        # h(x)  = tanh(x)
        # h'(x) = 1 - h(x)^2
        activation_func_derivatives = np.array(1 - np.square(self.outputs))

        # Calculate this layer's deltas (errors)
        # delta_j = h'(a_j)* sum(w_kj*d_k)
        # .T to transpose the weight matrix
        sum = np.matmul(deltas_next_layer, weights_to_next_layer.T)

        layer_deltas = activation_func_derivatives * sum

        return layer_deltas

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
