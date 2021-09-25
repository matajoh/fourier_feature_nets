import math
import numpy as np
import torch
import torch.nn as nn


class RawNeRF2d(nn.Module):
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


class BasicNeRF2d(nn.Module):
    def __init__(self, num_channels=256):
        nn.Module.__init__(self)
        self.input = nn.Linear(4, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, 3)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        encoded = 2 * math.pi * uv
        encoded = torch.cat([uv.cos(), uv.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        output = torch.sigmoid(self.output(output))
        return output


class PositionalNeRF2d(nn.Module):
    def __init__(self, num_channels=256, sigma=6, num_frequencies=256):
        nn.Module.__init__(self)
        frequencies = [2 * math.pi * math.pow(sigma, j / num_frequencies) for j in range(num_frequencies)]
        frequencies = torch.FloatTensor(frequencies)
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.input = nn.Linear(num_frequencies * 4, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, 3)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequencies.reshape(1, 1, -1)
        num_batch = uv.shape[0]
        frequencies = frequencies.expand(num_batch, -1, -1)
        encoded = torch.bmm(uv.reshape(-1, 2, 1), frequencies)
        encoded = encoded.reshape(num_batch, -1)
        encoded = torch.cat([encoded.cos(), encoded.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        output = torch.sigmoid(self.output(output))
        return output


class GaussianNeRF2d(nn.Module):
    def __init__(self, num_channels=256, sigma=10, num_frequencies=256):
        nn.Module.__init__(self)
        frequencies = 2 * math.pi * torch.normal(0, sigma, size=(2, num_frequencies))
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.input = nn.Linear(num_frequencies * 2, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, 3)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequencies.reshape(1, 2, -1)
        num_batch = uv.shape[0]
        frequencies = frequencies.expand(num_batch, -1, -1)
        encoded = torch.bmm(uv.reshape(-1, 1, 2), frequencies)
        encoded = encoded.reshape(num_batch, -1)
        encoded = torch.cat([encoded.cos(), encoded.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        output = torch.sigmoid(self.output(output))
        return output
