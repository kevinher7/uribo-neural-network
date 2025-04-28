import click

from src.model.create_model import create_neural_network


@click.command()
def console():
    """Entry point to access the Neural Network"""
    uribo_neural_network = create_neural_network()
    print(f"I am Uribo! This is me: {uribo_neural_network}")


if __name__ == "__main__":
    console()
