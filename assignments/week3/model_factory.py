import torch
from model import MLP


def create_model(input_dim: int, output_dim: int) -> MLP:
    """
    Create a multi-layer perceptron model.

    Arguments:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_dims (list): The dimensions of the hidden layers.

    Returns:
        MLP: The created model.

    """
    return MLP(input_dim, 520, output_dim, 1, torch.nn.ReLU(), torch.nn.init.ones_)

    # 1 hidden layer, optimal size = 520, accuracy = 97.85%
    # 2 hidden layer, optimal size = 257, accuracy = 97.56%
    # 3 hidden layer, optimal size = 256, dropout = 0.1, accuracy = 96.76%
