from src.model.neuron import Neuron
import numpy as np
from numpy.typing import NDArray


class Layer:
    def __init__(self, num_neurons: int) -> None:
        self.num_neurons = num_neurons
        self.input_dim = 0
        self.neurons = []

    def forward(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        self.outputs = []

        # neuronsリストに追加されたユニットを順に実行し、それぞれの結果をoutputsリストに追加
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(data))

        return np.array(self.outputs)

    def build(self) -> None:
        # neuronsリストにnum_nueronで指定した個数分ユニットを追加
        for index in range(self.num_neurons):
            self.neurons.append(Neuron(self.input_dim))
