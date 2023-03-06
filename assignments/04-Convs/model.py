import torch


class Model(torch.nn.Module):
    """
    build a neural network
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        construct model
        """
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
