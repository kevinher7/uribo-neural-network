import click
import numpy as np

from src.model.layer import Layer
from src.model.nn import NeuralNetwork


@click.command()
def console():
    """Entry point to access the Neural Network"""
    x_data = np.array([4])
    y_data = np.array([16])

    uribo_neural_network = NeuralNetwork(
        input_dim=x_data.size,
        layers=[
            Layer(3),
            Layer(2),
            Layer(1),
        ],
    )

    print(f"I am Uribo! This is me: {uribo_neural_network}")
    print("Let's train!!")
    print(f"Expected output; {y_data}")
    uribo_neural_network.train(x_data, y_data, epochs=20)
    print("I have finalized my training regime.")


if __name__ == "__main__":
    console()
