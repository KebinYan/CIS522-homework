import torch
from typing import Callable
import torch
import numpy as np


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        torch.manual_seed(522)
        super(MLP, self).__init__()
        self.activation = activation
        self.initializer = initializer

        self.layers = torch.nn.ModuleList()
        for i in range(hidden_count):
            output_size = hidden_size
            self.layers += [torch.nn.Linear(input_size, output_size, bias=True)]
            input_size = output_size
        self.out = torch.nn.Linear(input_size, num_classes, bias=True)
        self.dropout = torch.nn.Dropout(0.1)

        # print("hidden_layer: ", hidden_count, "hidden_size: ", hidden_size, "initializer: ", initializer)
        ...

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            #self.initializer(layer.weight)
            x = self.activation(layer(x))
            # x = self.dropout(x)

        x = self.out(x)
        return torch.nn.Softmax(dim=1)(x)
        ...
