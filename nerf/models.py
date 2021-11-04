"""Module containing various NeRF formulations."""

import math
import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import download_asset


class FourierFeatureMLP(nn.Module):
    """Version of NeRF using fourier features for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int,
                 frequencies_matrix: torch.Tensor, num_layers: int,
                 num_channels: int, output_sigmoid=False):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            frequencies_matrix (float): Frequency matrix
            num_layers (int): Number of layers in the MLP
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_sigmoid (bool, optional): Optional output sigmoid.
                                             Defaults to False.
        """
        nn.Module.__init__(self)
        self.params = {
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "frequencies_matrix": None if frequencies_matrix is None else frequencies_matrix.tolist(),
            "num_layers": num_layers,
            "num_channels": num_channels,
            "output_sigmoid": output_sigmoid
        }
        self.num_inputs = num_inputs
        self.output_act = torch.sigmoid if output_sigmoid else None
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
            encoded = (2 * math.pi * inputs) @ self.frequencies
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
        state_dict = self.state_dict()
        state_dict["type"] = "fourier"
        state_dict["params"] = self.params
        torch.save(state_dict, path)


class MLP(FourierFeatureMLP):
    """Unencoded FFN, essentially a standard MLP."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=4,
                 num_channels=256, output_sigmoid=False):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_sigmoid (bool, optional): Optional output sigmoid.
                                             Defaults to False.
        """
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   None, num_layers, num_channels,
                                   output_sigmoid)


class BasicFourierMLP(FourierFeatureMLP):
    """Basic version of FFN in which inputs are projected onto the unit circle."""

    def __init__(self, num_inputs: int, num_outputs: int, num_layers=4,
                 num_channels=256, output_sigmoid=False):
        """Constructor.

        Args:
            num_inputs (int): Number of dimensions in the input
            num_outputs (int): Number of dimensions in the output
            num_layers (int, optional): Number of layers in the MLP.
                                        Defaults to 4.
            num_channels (int, optional): Number of channels in the MLP.
                                          Defaults to 256.
            output_sigmoid (bool, optional): Optional output sigmoid.
                                             Defaults to False.
        """
        frequencies_matrix = torch.eye(num_inputs)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   frequencies_matrix, num_layers,
                                   num_channels, output_sigmoid)


class PositionalFourierMLP(FourierFeatureMLP):
    """Version of FFN with positional encoding."""
    def __init__(self, num_inputs: int, num_outputs: int, max_log_scale: float,
                 num_layers=4, num_channels=256, embedding_size=256,
                 output_sigmoid=False):
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
            output_sigmoid (bool, optional): Optional output sigmoid.
                                             Defaults to False.
        """
        frequencies_matrix = self.encoding(max_log_scale, embedding_size,
                                           num_inputs)
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs,
                                   frequencies_matrix, num_layers,
                                   num_channels, output_sigmoid)

    @staticmethod
    def encoding(max_log_scale: float, embedding_size: int, num_inputs: int):
        frequencies_matrix = [math.pow(max_log_scale, j / embedding_size)
                              for j in range(embedding_size)]
        frequencies_matrix = torch.FloatTensor(frequencies_matrix)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix


class GaussianFourierMLP(FourierFeatureMLP):
    """Version of a FFN using a full Gaussian matrix for encoding."""

    def __init__(self, num_inputs: int, num_outputs: int, sigma: float,
                 num_layers=4, num_channels=256, embedding_size=256,
                 output_sigmoid=False):
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
            output_sigmoid (bool, optional): Optional output sigmoid.
                                             Defaults to False.
        """
        frequencies = torch.normal(0, sigma, size=(num_inputs, embedding_size))
        FourierFeatureMLP.__init__(self, num_inputs, num_outputs, frequencies,
                                   num_layers, num_channels, output_sigmoid)


class NeRF(nn.Module):
    def __init__(self, num_layers: int, num_channels: int,
                 max_log_scale_pos: float, num_freq_pos: int,
                 max_log_scale_view: float, num_freq_view: int,
                 skips: Sequence[int], include_inputs: bool):
        nn.Module.__init__(self)
        self.params = {
            "num_layers": num_layers,
            "num_channels": num_channels,
            "max_log_scale_pos": max_log_scale_pos,
            "num_freq_pos": num_freq_pos,
            "max_log_scale_view": max_log_scale_view,
            "num_freq_view": num_freq_view,
            "skips": list(skips),
            "include_inputs": include_inputs
        }

        pos_encoding = self.encoding(max_log_scale_pos, num_freq_pos, 3)
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
        view_encoding = self.encoding(max_log_scale_view, num_freq_view, 3)
        self.view_encoding = nn.Parameter(view_encoding, requires_grad=False)
        self.skips = set(skips)
        self.include_inputs = include_inputs

        self.layers = nn.ModuleList()
        num_inputs = 2 * self.pos_encoding.shape[-1]
        if self.include_inputs:
            num_inputs += 3

        layer_inputs = num_inputs
        for i in range(num_layers):
            if i in self.skips:
                layer_inputs += num_inputs

            self.layers.append(nn.Linear(layer_inputs, num_channels))
            layer_inputs = num_channels

        self.opacity_out = nn.Linear(layer_inputs, 1)
        self.bottleneck = nn.Linear(layer_inputs, num_channels)

        layer_inputs = num_channels + 2 * self.view_encoding.shape[-1]
        if self.include_inputs:
            layer_inputs += 3

        self.hidden_view = nn.Linear(layer_inputs, num_channels // 2)
        layer_inputs = num_channels // 2
        self.color_out = nn.Linear(layer_inputs, 3)

    @staticmethod
    def encoding(max_log_scale: float, num_freq: int, num_inputs: int):
        frequencies_matrix = 2. ** torch.linspace(0, max_log_scale, num_freq)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix

    def forward(self, position, view):
        encoded_pos = position @ self.pos_encoding
        encoded_pos = [encoded_pos.cos(), encoded_pos.sin()]
        if self.include_inputs:
            encoded_pos.append(position)

        encoded_pos = torch.cat(encoded_pos, dim=-1)

        encoded_view = view @ self.view_encoding
        encoded_view = [encoded_view.cos(), encoded_view.sin()]
        if self.include_inputs:
            encoded_view.append(view)

        encoded_view = torch.cat(encoded_view, dim=-1)

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

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict["type"] = "nerf"
        state_dict["params"] = self.params
        torch.save(state_dict, path)


class Voxels(nn.Module):
    def __init__(self, side: int, scale: float):
        nn.Module.__init__(self)
        self.params = {
            "side": side,
            "scale": scale
        }

        voxels = torch.zeros((1, 4, side, side, side), dtype=torch.float32)
        self.voxels = nn.Parameter(voxels)
        bias = torch.zeros(4, dtype=torch.float32)
        bias[:3] = torch.logit(torch.FloatTensor([1e-5, 1e-5, 1e-5]))
        bias[3] = -2
        self.bias = nn.Parameter(bias.unsqueeze(0))
        self.scale = scale

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.reshape(1, -1, 1, 1, 3)
        positions = positions / self.scale
        output = F.grid_sample(self.voxels, positions,
                               padding_mode="border", align_corners=False)
        output = output.transpose(1, 2)
        output = output.reshape(-1, 4)
        output = output + self.bias
        return output

    def save(self, path: str):
        """Saves the model to the specified path.

        Args:
            path (str): Path to the model file on disk
        """
        state_dict = self.state_dict()
        state_dict["type"] = "voxels"
        state_dict["params"] = self.params
        torch.save(state_dict, path)


def load_model(path: str):
    if not os.path.exists(path):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        path = os.path.join(models_dir, path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print("Downloading model...")
            model_name = os.path.basename(path)
            success = download_asset(model_name, path)
            if not success:
                print("Unable to download model", model_name)
                return None

    state_dict = torch.load(path)
    if state_dict["type"] == "fourier":
        model_class = FourierFeatureMLP
    elif state_dict["type"] == "nerf":
        model_class = NeRF
    elif state_dict["type"] == "voxels":
        model_class = Voxels
    else:
        print("Unrecognized model type:", state_dict["type"])

    del state_dict["type"]
    model = model_class(**state_dict["params"])
    del state_dict["params"]
    model.load_state_dict(state_dict)
    model.eval()
    return model
