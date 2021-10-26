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
        frequencies_matrix = self.encoding(sigma, num_frequencies, num_inputs)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   frequencies_matrix, num_layers,
                                   num_channels, output_act)

    @staticmethod
    def encoding(sigma: float, num_frequencies: int, num_inputs: int):
        num_steps = num_frequencies // num_inputs
        frequencies_matrix = 2 ** torch.linspace(0, sigma * 2 * math.pi, num_steps) - 1
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix


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


class NeRF(nn.Module):
    def __init__(self, num_layers=8, num_channels=256,
                 sigma_pos=8, num_freq_pos=8,
                 sigma_view=4, num_freq_view=4,
                 skips=(4)):
        self.pos_encoding = PositionalFourierMLP.encoding(sigma_pos, num_freq_pos, 3)
        self.view_encoding = PositionalFourierMLP.encoding(sigma_view, num_freq_view, 3)
        self.skips = set(skips)
        
        self.layers = nn.ModuleList()
        num_inputs = self.pos_encoding.shape[-1]
        
        layer_inputs = num_inputs
        for i in range(num_layers):
            if i in self.skips:
                layer_inputs += num_inputs

            self.layers.append(nn.Linear(layer_inputs, num_channels))
            layer_inputs = num_channels

        self.opacity_out = nn.Linear(layer_inputs, 1)
        self.bottleneck = nn.Linear(layer_inputs, num_channels)

        layer_inputs = num_channels + self.view_encoding.shape[-1]
        self.hidden_view = nn.Linear(layer_inputs, num_channels // 2)
        layer_inputs = num_channels // 2
        self.color_out = nn.Linear(layer_inputs, 3)

    def forward(self, position, view):
        pos = position @ self.pos_encoding
        encoded_pos = torch.cat([pos.cos(), pos.sin()], dim=-1)

        view = view @ self.view_encoding
        encoded_view = torch.cat([view.cos(), view.sin()], dim=-1)

        outputs = encoded_pos
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                outputs = torch.cat([outputs, encoded_pos], dim=-1)

            outputs = torch.relu(layer(outputs))

        opacity = self.opacity_out(outputs)
        bottleneck = self.bottleneck(outputs)

        outputs = torch.cat([bottleneck, encoded_view], dim=-1)
        outputs = torch.relu(self.hidden_view(outputs))
        color = self.color_out(outputs)
        return torch.cat([color, opacity], dim=-1)
