from src.model.neuron import Neuron
import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self, num_neurons: int, *, output_layer: bool = False) -> None:
        self.num_neurons = num_neurons

        self.input_dim = 0
        self.output_layer = output_layer

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.outputs = []
        activations = np.matmul(data, self.weights)

        # Following a dense architecture (exactly 1 activation per neuron)
        for index in range(len(activations)):
            self.outputs.append(self.neurons[index].activate(activations[index]))

        return np.array(self.outputs)

    def backward(
        self,
        delta_next_layer: NDArray[np.float64],
        weights_to_next_layer: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        for i in range(len(self.neurons) - 1, 0, -1):
            self.neurons[i].backward(delta_next_layer, weights_to_next_layer)

    def build(self, input_dim) -> None:
        self.weights = np.random.randn(input_dim, self.num_neurons)
        self.neurons: list[Neuron] = [Neuron() for _index in range(self.num_neurons)]
