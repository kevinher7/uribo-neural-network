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

        activations = np.matmul(data, self.weights)
        self.outputs = self._activation_function(activations)

        if self.augmentation:
            self.outputs = np.append(self.outputs, 1.0)

        return self.outputs

    def backward(
        self,
        deltas_next_layer: NDArray[np.float64],
        weights_to_next_layer: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Calculate the gradient
        # grad_j = d_k*z_j
        print("delta", deltas_next_layer)
        print("outputs", self.outputs)
        self.grad = np.matmul(deltas_next_layer, self.outputs.reshape(1, -1))

        # Since we use hyperbolic tangent as the activation function
        # h(x)  = tanh(x)
        # h'(x) = 1 - h(x)^2
        activation_func_derivatives = np.array(1 - np.square(self.outputs))

        # This layer's deltas (errors)
        # delta_j = h'(a_j)* sum(w_kj*d_k)
        print("deltas", deltas_next_layer)
        print("weights", weights_to_next_layer)
        sum = np.matmul(deltas_next_layer, weights_to_next_layer.reshape(1, -1))
        print("sum", sum)
        print("activation_func_derivatives", activation_func_derivatives)
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
