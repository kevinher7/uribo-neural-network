from src.model.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, data: list[int], classification: bool = False) -> None:
        """
        Forward method for layer class.
        It takes a one dimensional numpy array as input.
        """
        layer_output = data
        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        if classification:
            output = self._softmax(layer_output)
        else:
            output = layer_output

        return output

    def _softmax() -> None:
        raise NotImplementedError(
            "Neural Network does not support classification (only regression)"
        )
