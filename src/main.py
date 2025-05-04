import click

from src.model.layer import Layer
from src.model.nn import NeuralNetwork


@click.command()
def console():
    """Entry point to access the Neural Network"""
    uribo_neural_network = NeuralNetwork(
        input_dim=3,
        layers=[
            Layer(3),
            Layer(1),
        ],
    )

    print(f"I am Uribo! This is me: {uribo_neural_network}")


if __name__ == "__main__":
    console()
