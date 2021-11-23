"""Module containing the full NeRF model."""

from typing import Sequence

import torch
import torch.nn as nn


class NeRF(nn.Module):
    """The full NeRF model."""

    def __init__(self, num_layers: int, num_channels: int,
                 max_log_scale_pos: float, num_freq_pos: int,
                 max_log_scale_view: float, num_freq_view: int,
                 skips: Sequence[int], include_inputs: bool):
        """Constructor.

        Args:
            num_layers (int): Number of layers in the main body.
            num_channels (int): Number of channels per layer.
            max_log_scale_pos (float): The maximum log scale for the positional
                                       encoding.
            num_freq_pos (int): The number of frequences to use for encoding
                                position.
            max_log_scale_view (float): The maximum log scale for the view
                                        direction.
            num_freq_view (int): The number of frequencies to use for encoding
                                 view direction.
            skips (Sequence[int]): Skip connection layers.
            include_inputs (bool): Whether to include the inputs in the encoded
                                   input vector.
        """
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

        pos_encoding = self._encoding(max_log_scale_pos, num_freq_pos, 3)
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
        view_encoding = self._encoding(max_log_scale_view, num_freq_view, 3)
        self.view_encoding = nn.Parameter(view_encoding, requires_grad=False)
        self.skips = set(skips)
        self.include_inputs = include_inputs
        self.use_view = True

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
    def _encoding(max_log_scale: float, num_freq: int, num_inputs: int):
        frequencies_matrix = 2. ** torch.linspace(0, max_log_scale, num_freq)
        frequencies_matrix = frequencies_matrix.reshape(-1, 1, 1)
        frequencies_matrix = torch.eye(num_inputs) * frequencies_matrix
        frequencies_matrix = frequencies_matrix.reshape(-1, num_inputs)
        frequencies_matrix = frequencies_matrix.transpose(0, 1)
        return frequencies_matrix

    def forward(self, position: torch.Tensor,
                view: torch.Tensor) -> torch.Tensor:
        """Queries the model for the radiance field output.

        Args:
            position (torch.Tensor): a (N,3) tensor of positions.
            view (torch.Tensor): a (N,3) tensor of normalized view directions.

        Returns:
            torch.Tensor: a (N,4) tensor of color and opacity.
        """
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
