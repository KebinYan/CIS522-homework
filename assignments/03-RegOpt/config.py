from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 10
    initial_learning_rate = 1e-3
    initial_weight_decay = 0
    momentum = 0.9

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "lr_lambdas": lambda epoch: epoch
        / 10,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
