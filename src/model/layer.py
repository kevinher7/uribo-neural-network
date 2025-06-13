from src.model.neuron import Neuron
import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self, num_neurons: int, *, output_layer: bool = False) -> None:
        self.num_neurons = num_neurons
        self.neurons: list[Neuron] = [Neuron() for _index in range(num_neurons)]

        self.input_dim = 0
        self.output_layer = output_layer

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.outputs = []

        # neuronsリストに追加されたユニットを順に実行し、それぞれの結果をoutputsリストに追加
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(data))

        return np.array(self.outputs)

    def backward(
        self,
        delta_next_layer: NDArray[np.float64],
        weights_to_next_layer: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        for i in range(len(self.neurons) - 1, 0, -1):
            self.neurons[i].backward(delta_next_layer, weights_to_next_layer)

    def build(self) -> None:
        self.weights = np.random.randn(self.input_dim, self.num_neurons)

        if self.output_layer:
            for _index in range(self.num_neurons):
                self.neurons.append(Neuron(self.input_dim))

        self.weights = [neuron.weight for neuron in self.neurons]

        return

        # Add "bias neuron" to neuron list
        self.neurons.append(Neuron(self.input_dim, bias_neuron=True))

        # Add remaining neurons
        for _index in range(self.num_neurons - 1):
            self.neurons.append(Neuron(self.input_dim))
