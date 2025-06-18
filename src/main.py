import click
import numpy as np

from src.model.layer import Layer
from src.model.nn import NeuralNetwork

# Set random seed for reproducibility
np.random.seed(42)


@click.command()
def console():
    """Entry point to access the Neural Network"""
    # Generate parabola data
    x_raw = np.array([[x] for x in range(-10, 11)], dtype=np.float64)
    y_raw = np.array([x**2 for x in range(-10, 11)], dtype=np.float64)

    # Normalize inputs to [-1, 1] range for better training
    x_data = x_raw / 10.0
    y_data = y_raw / 100.0

    uribo_neural_network = NeuralNetwork(
        input_dim=x_data[0].size,
        layers=[
            Layer(3),
            Layer(2),
            Layer(1),
        ],
    )

    print(f"I am Uribo! This is me: {uribo_neural_network}")
    print("Let's train!!")
    uribo_neural_network.train(x_data, y_data, epochs=1000)
    print("I have finalized my training regime.")


if __name__ == "__main__":
    console()
