"""Module providing a simple voxel-based radiance field model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Voxels(nn.Module):
    """A voxel based radiance field model."""

    def __init__(self, side: int, scale: float):
        """Constructor.

        Args:
            side (int): The number of voxels on one side of a cube.
            scale (float): The scale of the voxel volume, equivalent
                           to half of one side of the volume, i.e. a
                           scale of 1 indicates a volume of size 2x2x2.
        """
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
        self.use_view = False

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Interpolates the positions within the voxel volume."""
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
