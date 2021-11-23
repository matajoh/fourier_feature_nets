"""Module containing models based upon the Fourier Feature Networks template."""

import math
from typing import List

import torch
import torch.nn as nn


class FourierFeatureMLP(nn.Module):
    """MLP which uses Fourier features as a preprocessing step."""

    def __init__(self, num_inputs: int, num_outputs: int,
                 a_values: torch.Tensor, b_values: torch.Tensor,
                 layer_channels: List[int]):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            a_values (torch.Tensor): a values for encoding
            b_values (torch.Tensor): b values for encoding
            num_layers (int): Number of layers in the MLP
            layer_channels (List[int]): Number of channels per layer.
        """
        nn.Module.__init__(self)
        self.params = {
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "a_values": None if a_values is None else a_values.tolist(),
            "b_values": None if b_values is None else b_values.tolist(),
            "layer_channels": layer_channels
        }
        self.num_inputs = num_inputs
        if b_values is None:
            self.a_values = None
            self.b_values = None
            num_inputs = num_inputs
        else:
            assert b_values.shape[0] == num_inputs
            assert a_values.shape[0] == b_values.shape[1]
            self.a_values = nn.Parameter(a_values, requires_grad=False)
            self.b_values = nn.Parameter(b_values, requires_grad=False)
            num_inputs = b_values.shape[1] * 2

        self.layers = nn.ModuleList()
        for num_channels in layer_channels:
            self.layers.append(nn.Linear(num_inputs, num_channels))
            num_inputs = num_channels

        self.layers.append(nn.Linear(num_inputs, num_outputs))

        self.use_view = False
        self.keep_activations = False
        self.activations = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predicts outputs from the provided uv input."""
        if self.b_values is None:
            output = inputs
        else:
            # NB: the below should be 2*math.pi, but the values
            # coming in are already in the range of -1 to 1 or
            # 0 to 2, so we want to keep the range so that it does
            # not exceed 2pi
            encoded = (math.pi * inputs) @ self.b_values
            output = torch.cat([self.a_values * encoded.cos(),
                                self.a_values * encoded.sin()], dim=-1)

        self.activations.clear()
        for layer in self.layers[:-1]:
            output = torch.relu(layer(output))

        if self.keep_activations:
            self.activations.append(output.detach().cpu().numpy())

        output = self.layers[-1](output)
        return output

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict["type"] = "fourier"
        state_dict["params"] = self.params
        torch.save(state_dict, path)


class MLP(FourierFeatureMLP):
    """Unencoded FFN, essentially a standard MLP."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=3,
                 num_channels=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   None, None,
                                   [num_channels] * num_layers)


class BasicFourierMLP(FourierFeatureMLP):
    """Basic version of FFN in which inputs are projected onto the unit circle."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=3,
                 num_channels=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
        """
        a_values = torch.ones(num_inputs)
        b_values = torch.eye(num_inputs)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   a_values, b_values,
                                   [num_channels] * num_layers)


class PositionalFourierMLP(FourierFeatureMLP):
    """Version of FFN with positional encoding."""
    def __init__(self, num_inputs: int, num_outputs: int, max_log_scale: float,
                 num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            max_log_scale (float): Maximum log scale for embedding
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): The size of the feature embedding.
                                            Defaults to 256.
        """
        b_values = self._encoding(max_log_scale, embedding_size, num_inputs)
        a_values = torch.ones(b_values.shape[1])
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   a_values, b_values,
                                   [num_channels] * num_layers)

    @staticmethod
    def _encoding(max_log_scale: float, embedding_size: int, num_inputs: int):
        """Produces the encoding b_values matrix."""
        embedding_size = embedding_size // num_inputs
        frequencies_matrix = 2. ** torch.linspace(0, max_log_scale, embedding_size)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix


class GaussianFourierMLP(FourierFeatureMLP):
    """Version of a FFN using a full Gaussian matrix for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_layers=3, num_channels=256, embedding_size=256):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            sigma (float): Standard deviation of the Gaussian distribution
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            embedding_size (int, optional): Number of frequencies to use for
                                             the encoding. Defaults to 256.
        """
        b_values = torch.normal(0, sigma, size=(num_inputs, embedding_size))
        a_values = torch.ones(b_values.shape[1])
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   a_values, b_values,
                                   [num_channels] * num_layers)
