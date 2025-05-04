from neuron import Neuron
import numpy as np


class Layer:
    def __init__(self, num_neuron: int, input_dim) -> None:
        self.neurons = []
        # neuronsリストにnum_nueronで指定した個数分ユニットを追加
        for index in range(num_neuron):
            self.neurons.append(Neuron(input_dim))

    def forward(self, data):
        self.outputs = []
        # neuronsリストに追加されたユニットを順に実行し、それぞれの結果をoutputsリストに追加
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(data))
        return np.array(self.outputs)
