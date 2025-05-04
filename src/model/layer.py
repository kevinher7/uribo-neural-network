from neuron import Neuron
import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self, num_neuron: int, input_dim: int) -> None:
        self.neurons = []
        # neuronsリストにnum_nueronで指定した個数分ユニットを追加
        for index in range(num_neuron):
            self.neurons.append(Neuron(input_dim))

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.outputs = []
        # neuronsリストに追加されたユニットを順に実行し、それぞれの結果をoutputsリストに追加
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(data))
        return np.array(self.outputs)
