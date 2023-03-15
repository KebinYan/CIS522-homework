import torch


class Model(torch.nn.Module):
    """
    build a neural network
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(5, 5)
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 10)
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        construct model
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
