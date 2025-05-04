from src.model.neuron import Neuron


class Layer:
    def __init__(self, num_neuron: int, input_dim) -> None:
        self.neurons = []
        for index in num_neuron:
            self.neurons.append(Neuron(input_dim))

    def forward(self, data):
        self.outputs = []
        for neuron in self.neurons:
            self.outputs.append(neuron.forward(data))
