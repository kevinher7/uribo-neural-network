import click
import numpy as np

from src.model.layer import Layer
from src.model.nn import NeuralNetwork


@click.command()
def console():
    """Entry point to access the Neural Network"""
    x_data = np.array([1])

    uribo_neural_network = NeuralNetwork(
        input_dim=x_data.size,
        layers=[
            Layer(3),
            Layer(1),
        ],
    )

    print(f"I am Uribo! This is me: {uribo_neural_network}")
    print(f"This is my output with for x={x_data}: {uribo_neural_network.forward(x_data)}")


if __name__ == "__main__":
    console()
