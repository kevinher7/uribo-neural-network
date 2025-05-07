from src.model.neuron import Neuron
import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self, num_neurons: int, *, output_layer: bool = False) -> None:
        self.num_neurons = num_neurons
        self.input_dim = 0
        self.neurons = []
        self.output_layer = output_layer

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.outputs = []

        # neuronsリストに追加されたユニットを順に実行し、それぞれの結果をoutputsリストに追加
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(data))

        return np.array(self.outputs)

    def build(self) -> None:
        # neuronsリストにnum_nueronで指定した個数分ユニットを追加
        if self.output_layer:
            for _index in range(self.num_neurons):
                self.neurons.append(Neuron(self.input_dim))

            return

        # Add "bias neuron" to neuron list
        self.neurons.append(Neuron(self.input_dim, bias_neuron=True))

        # Add remaining neurons
        for _index in range(self.num_neurons - 1):
            self.neurons.append(Neuron(self.input_dim))
