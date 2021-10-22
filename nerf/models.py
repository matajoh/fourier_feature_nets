"""Module containing various NeRF formulations."""

import math

import torch
import torch.nn as nn


class FourierFeatureMLP(nn.Module):
    """Version of NeRF using fourier features for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int,
                 frequencies_matrix: torch.Tensor, num_layers: int,
                 num_channels: int, output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            frequencies_matrix (float): Frequency matrix
            num_layers (int): Number of layers in the MLP
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.output_act = output_act
        if frequencies_matrix is None:
            self.frequencies = None
            num_inputs = num_inputs
        else:
            assert frequencies_matrix.shape[0] == num_inputs
            self.frequencies = nn.Parameter(frequencies_matrix,
                                            requires_grad=False)
            num_inputs = frequencies_matrix.shape[1] * 2

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_inputs, num_channels))
            layers.append(nn.ReLU())
            num_inputs = num_channels

        layers.append(nn.Linear(num_inputs, num_outputs))

        for layer in layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predicts outputs from the provided uv input."""
        if self.frequencies is None:
            output = inputs
        else:
            encoded = inputs @ self.frequencies
            output = torch.cat([encoded.cos(), encoded.sin()], dim=-1)

        output = self.layers(output)

        if self.output_act is None:
            return output

        return self.output_act(output)

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


class MLP(FourierFeatureMLP):
    """Unencoded FFN, essentially a standard MLP."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=4,
                 num_channels=256, output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   None, num_layers, num_channels, output_act)


class BasicFourierMLP(FourierFeatureMLP):
    """Basic version of FFN in which inputs are projected onto the unit circle."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=4,
                 num_channels=256, output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        frequencies_matrix = torch.eye(num_inputs) * 2 * math.pi
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   frequencies_matrix, num_layers,
                                   num_channels, output_act)


class PositionalFourierMLP(FourierFeatureMLP):
    """Version of FFN with positional encoding."""
    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_layers=4, num_channels=256, num_frequencies=256,
                 output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Maximum log scale
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            num_frequencies (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        num_steps = num_frequencies // num_inputs
        frequencies_matrix = 2 ** torch.linspace(0, sigma * 2 * math.pi, num_steps) - 1
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   frequencies_matrix, num_layers,
                                   num_channels, output_act)


class GaussianFourierMLP(FourierFeatureMLP):
    """Version of a FFN using a full Gaussian matrix for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_layers=4, num_channels=256, num_frequencies=256,
                 output_act=None):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Standard deviation of normal distribution used for
                           sampling the Fourier exponents.
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            num_frequencies (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
            output_act (Callable, optional): Optional output activation.
                                             Defaults to None.
        """
        frequencies = torch.normal(0, sigma, size=(num_inputs, num_frequencies))
        frequencies *= 2 * math.pi
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs, frequencies,
                                   num_layers, num_channels, output_act)
