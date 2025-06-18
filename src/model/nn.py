import numpy as np
from numpy.typing import NDArray

from src.model.layer import Layer


class NeuralNetwork:
    def __init__(self, input_dim: int, layers: list[Layer], *, augmentation: bool = True) -> None:
        """Initialize NN by assigning input dimensions and building all Layers"""
        self.augmentation = augmentation

        self.layers: list[Layer] = []
        for layer in layers:
            layer.build(input_dim, augmentation)
            self.layers.append(layer)
            input_dim = layer.num_neurons

        # Set last layer as output layer
        self.layers[-1].is_output = True

    def train(
        self,
        data: NDArray[np.float64],
        targets: NDArray[np.float64],
        *,
        epochs: int,
        learning_rate: float = 0.01,
    ) -> NDArray[np.float64]:
        for epoch in range(epochs):
            total_loss = 0.0
            for index, datapoint in enumerate(data):
                if self.augmentation:
                    datapoint = np.append(datapoint, 1)

                output = self._forward(datapoint)
                loss = 0.5 * (output - targets[index]) ** 2  # MSE loss
                total_loss += loss
                self._backward(datapoint, targets[index], output, learning_rate)

            # Print loss every 50 epochs
            if epoch % 50 == 0:
                avg_loss = total_loss / len(data)
                print(f"Epoch {epoch}, Average Loss: {avg_loss}")

        # Test on some sample points
        print("\nFinal predictions:")
        for i in [0, 10, 20]:  # Test on x = -10, 0, 10 (normalized)
            if i < len(data):
                test_input = np.append(data[i], 1) if self.augmentation else data[i]
                prediction = self._forward(test_input)
                print(
                    f"x = {data[i][0] * 10:.1f}, predicted y = {prediction * 100}, actual y = {targets[i] * 100:.2f}"
                )

    def _forward(self, data: NDArray[np.float64], *, classification: bool = False) -> float:
        """
        Forward method for layer class.
        Args:
            data (NDArray[np.float64]): 1 dimmensional numpy array

        Kwargs:
            classification (bool): determines whether to use regression or classification

        Returns:
            output (float): for regression
        """
        # Understood as the "input layer" output
        layer_output = data

        for layer in self.layers:
            layer_output = layer.forward(layer_output)

        if classification:
            return self._softmax(layer_output)
        else:
            return layer_output

    def _backward(
        self,
        input_data: NDArray[np.float64],
        target_data: NDArray[np.float64],
        model_output: NDArray[np.float64],
        learning_rate: float,
    ) -> None:
        """
        Backward Propagation to calculate gradients and update weights
        """
        # Calculate delta for output layer (derivative of MSE loss)
        delta = model_output - target_data

        # Backpropagate through all layers in reverse order
        for i in reversed(range(len(self.layers))):
            # Calculate gradient for this layer
            gradient = np.outer(delta, self.layers[i].inputs)

            # Update weights for this layer
            self.layers[i].weights -= learning_rate * gradient.T

            # Calculate delta for previous layer (if not the first layer)
            if i > 0:
                # Remove bias weight if augmentation is used
                weights_no_bias = (
                    self.layers[i].weights[:-1] if self.augmentation else self.layers[i].weights
                )

                # Propagate delta back
                delta = np.dot(delta, weights_no_bias.T)

                prev_outputs = self.layers[i - 1].outputs
                if self.augmentation:
                    # Get the outputs of the previous layer (without bias)
                    prev_outputs = (
                        prev_outputs[:-1] if len(prev_outputs.shape) == 1 else prev_outputs
                    )

                # For tanh: derivative is 1 - tanh^2(x)
                delta = delta * (1 - prev_outputs**2)

    def _softmax() -> None:
        raise NotImplementedError(
            "Uribo Neural Network does not support classification (only regression)"
        )

    # Note: Weight updates are now integrated into the _backward method for efficiency
