"""Module containing various NeRF formulations."""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Class representing a version of NeRF without any positional encoding."""
    def __init__(self, num_inputs: int, num_outputs: int, num_channels=256,
                 output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        nn.Module.__init__(self)
        self.output_act = output_act
        self.input = nn.Linear(num_inputs, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, num_outputs)
        for layer in [self.input, self.hidden0, self.hidden1, self.output]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Predicts outputs from the provided uv input."""
        output = torch.relu(self.input(uv))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        raw_output = self.output(output)
        if self.output_act is None:
            return raw_output

        return self.output_act(raw_output)

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the model from the provided path.

        Args:
            path (str): Path to the model file on disk
        """
        self.load_state_dict(torch.load(path))
        self.eval()


class BasicFourierMLP(nn.Module):
    """Basic version of NeRF in which inputs are projected onto the unit circle."""

    def __init__(self, num_inputs: int, num_outputs: int, num_channels=256,
                 output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        nn.Module.__init__(self)
        self.output_act = output_act
        self.input = nn.Linear(num_inputs * 2, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, num_outputs)
        for layer in [self.input, self.hidden0, self.hidden1, self.output]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Predicts color from the provided uv input."""
        encoded = 2 * math.pi * uv
        encoded = torch.cat([uv.cos(), uv.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        raw_output = self.output(output)
        if self.output_act is None:
            return raw_output

        return self.output_act(raw_output)

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the model from the provided path.

        Args:
            path (str): Path to the model file on disk
        """
        self.load_state_dict(torch.load(path))
        self.eval()


class PositionalFourierMLP(nn.Module):
    """Version of NeRF with positional encoding."""
    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_channels=256, num_frequencies=256, output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Fourier exponent.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            num_frequencies (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        nn.Module.__init__(self)
        frequencies = [2 * math.pi * math.pow(sigma, j / num_frequencies) for j in range(num_frequencies)]
        frequencies = torch.FloatTensor(frequencies)
        self.num_inputs = num_inputs
        self.output_act = output_act
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.input = nn.Linear(num_frequencies * 2 * num_inputs, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, num_outputs)
        for layer in [self.input, self.hidden0, self.hidden1, self.output]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Predicts outputs from the provided uv input."""
        frequencies = self.frequencies.reshape(1, 1, -1)
        num_batch = uv.shape[0]
        frequencies = frequencies.expand(num_batch, -1, -1)
        encoded = torch.bmm(uv.reshape(-1, self.num_inputs, 1), frequencies)
        encoded = encoded.reshape(num_batch, -1)
        encoded = torch.cat([encoded.cos(), encoded.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        raw_output = self.output(output)
        if self.output_act is None:
            return raw_output

        return self.output_act(raw_output)

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the model from the provided path.

        Args:
            path (str): Path to the model file on disk
        """
        self.load_state_dict(torch.load(path))
        self.eval()


class GaussianFourierMLP(nn.Module):
    """Version of NeRF using a full Gaussian matrix for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_channels=256, num_frequencies=256, output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Standard deviation of normal distribution used for
                           sampling the Fourier exponents.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            num_frequencies (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.output_act = output_act
        frequencies = torch.normal(0, sigma, size=(num_inputs, num_frequencies))
        frequencies *= 2 * math.pi
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.input = nn.Linear(num_frequencies * 2, num_channels)
        self.hidden0 = nn.Linear(num_channels, num_channels)
        self.hidden1 = nn.Linear(num_channels, num_channels)
        self.output = nn.Linear(num_channels, num_outputs)
        for layer in [self.input, self.hidden0, self.hidden1, self.output]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Predicts outputs from the provided uv input."""
        frequencies = self.frequencies.reshape(1, self.num_inputs, -1)
        num_batch = uv.shape[0]
        frequencies = frequencies.expand(num_batch, -1, -1)
        encoded = torch.bmm(uv.reshape(-1, 1, self.num_inputs), frequencies)
        encoded = encoded.reshape(num_batch, -1)
        encoded = torch.cat([encoded.cos(), encoded.sin()], dim=-1)
        output = torch.relu(self.input(encoded))
        output = torch.relu(self.hidden0(output))
        output = torch.relu(self.hidden1(output))
        raw_output = self.output(output)
        if self.output_act is None:
            return raw_output

        return self.output_act(raw_output)

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Loads the model from the provided path.

        Args:
            path (str): Path to the model file on disk
        """
        self.load_state_dict(torch.load(path))
        self.eval()
