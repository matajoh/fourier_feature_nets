import torch
import torch.nn as nn


class RawNerf2d(nn.Module):
    def __init__(self, num_channels=256):
        nn.Module.__init__(self)
        self.input = nn.Linear(2, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, 3)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        output = torch.relu(self.input(uv))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        output = torch.sigmoid(self.output(output))
        return output


class BasicNerf2d(nn.Module):
    def __init__(self, num_channels=256):
        nn.Module.__init__(self)
        self.input = nn.Linear(4, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, 3)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat([uv.cos(), uv.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        output = torch.sigmoid(self.output(output))
        return output
