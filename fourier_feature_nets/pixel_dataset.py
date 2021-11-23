"""Module containing logic for pixel prediction dataset."""

import math
import os
from typing import NamedTuple

import cv2
import numpy as np
import torch

from .fourier_feature_models import FourierFeatureMLP


class PixelData(NamedTuple("PixelData", [("uv", torch.FloatTensor),
                                         ("color", torch.FloatTensor)])):
    """Tuple representing pixel data.

    Description:
        There are two components:

        uv (torch.Tensor): a Nx2 tensor of UV values ranging from -1 to 1
        color (torch.Tensor): a Nx3 tensor of color values
    """


class PixelDataset:
    """Dataset consisting of image pixels."""

    def __init__(self, size: int, color_space: str,
                 train_data: PixelData, val_data: PixelData):
        """Constructor.

        Args:
            size (int): Square size of the image
            color_space (str): Color space used ("RGB" or "YCrCb")
            train_data (PixelData): The training data tensors
            val_data (PixelData): The validation data tensors
        """
        self.size = size
        self.color_space = color_space
        self.image = self.to_image(val_data.color)
        self.train_uv, self.train_color = train_data
        self.val_uv, self.val_color = val_data

    @staticmethod
    def create(path: str, color_space: str, size=512) -> "PixelDataset":
        """Creates a dataset from an image.

        Args:
            path (str): the path to the image file
            color_space (str): Color space to use ("RGB" or "YCrCb")
            size (int, optional): Size to use when resizing the image.
                                  Defaults to 512.

        Raises:
            NotImplementedError: Raised if provided an unsupported color space

        Returns:
            PixelDataset: the constructed dataset
        """
        if not os.path.exists(path):
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            path = os.path.join(data_dir, path)
            path = os.path.abspath(path)

        pixels = cv2.imread(path)
        if pixels is None:
            print("Unable to load image at", path)
            return None

        if pixels.shape[0] > pixels.shape[1]:
            start = (pixels.shape[0] - pixels.shape[1]) // 2
            end = start + pixels.shape[1]
            pixels = pixels[start:end, :]
        elif pixels.shape[1] > pixels.shape[0]:
            start = (pixels.shape[1] - pixels.shape[0]) // 2
            end = start + pixels.shape[0]
            pixels = pixels[:, start:end]

        if pixels.shape[0] != size:
            pixels = cv2.resize(pixels, (size, size), cv2.INTER_AREA)

        if color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb) / 255
        elif color_space == "RGB":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB) / 255
        else:
            raise NotImplementedError("Unsupported color space: {}".format(color_space))

        # NB this slightly unorthodox uv range of 0 to 2 is due to the FFN
        # expecting its input values to have a range that encompasses
        # 2 (as is the case in 3D volume modeling).

        vals = np.linspace(0, 2, size // 2, endpoint=False, dtype=np.float32)
        train_uv = np.stack(np.meshgrid(vals, vals), axis=-1)
        train_color = pixels[::2, ::2, :]

        vals = np.linspace(0, 2, size, endpoint=False, dtype=np.float32)
        val_uv = np.stack(np.meshgrid(vals, vals), axis=-1)
        val_color = pixels

        train_data = PixelData(torch.from_numpy(train_uv), torch.from_numpy(train_color))
        val_data = PixelData(torch.from_numpy(val_uv), torch.from_numpy(val_color))
        return PixelDataset(size, color_space, train_data, val_data)

    def to(self, *args) -> "PixelDataset":
        """Equivalent of torch.Tensor.to for all tensors in the dataset.

        Returns:
            PixelDataset: the result of the to operation
        """
        train_data = PixelData(self.train_uv.to(*args), self.train_color.to(*args))
        val_data = PixelData(self.val_uv.to(*args), self.val_color.to(*args))
        return PixelDataset(self.size, self.color_space, train_data, val_data)

    def to_act_image(self, model: FourierFeatureMLP, size: int) -> np.ndarray:
        """Produces a grid image of activations."""
        num_grid = 8
        grid_size = size // num_grid
        uvs = self.generate_uvs(grid_size, next(model.parameters()).device)
        uvs = uvs.reshape(-1, 2)
        model.keep_activations = True
        with torch.no_grad():
            model(uvs)

        model.keep_activations = False

        palette = model.layers[-1].weight.data.detach().cpu().numpy()
        bias = model.layers[-1].bias.data.detach().cpu().numpy()
        activation = model.activations[-1].T
        activation = activation[..., np.newaxis]
        palette = palette.T[:, np.newaxis, :]
        activation_values = torch.from_numpy(activation * palette + bias)
        activation_values = torch.sigmoid(activation_values).numpy()
        index = np.arange(num_grid*num_grid)
        act_pixels = np.zeros((size, size, 3), np.float32)
        for i in range(num_grid):
            rstart = i * grid_size
            rend = rstart + grid_size
            for j in range(num_grid):
                cstart = j * grid_size
                cend = cstart + grid_size
                values = activation_values[index[i*num_grid + j]]
                values = values.reshape(grid_size, grid_size, 3)
                act_pixels[rstart:rend, cstart:cend] = values

        act_pixels = (act_pixels * 255).astype(np.uint8)
        if self.color_space == "YCrCb":
            act_pixels = cv2.cvtColor(act_pixels, cv2.COLOR_YCrCb2RGB)

        return act_pixels

    def to_image(self, colors: torch.Tensor, size=0) -> np.ndarray:
        """Converts predicted colors back into an image.

        Args:
            colors (torch.Tensor): The predicted colors
            size (int, optional): The desired size
                                  (if different from the dataset).
                                  Defaults to 0.

        Returns:
            np.ndarray: the image pixels in BGR format
        """
        if size == 0:
            size = self.size

        pixels = (colors * 255).reshape(size, size, 3).cpu().numpy().astype(np.uint8)
        if self.color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCb2RGB)

        return pixels

    @staticmethod
    def generate_uvs(size: int, device) -> torch.Tensor:
        """Generates UV values for the specified size.

        Args:
            size (int): Image size to use when computing UVs.
            device: The torch device to use when creating the Tensor

        Returns:
            torch.Tensor: The image UVs
        """
        vals = np.linspace(0, 2, size, endpoint=False, dtype=np.float32)
        uvs = np.stack(np.meshgrid(vals, vals), axis=-1)
        return torch.from_numpy(uvs).to(device=device)

    def psnr(self, colors: torch.Tensor) -> float:
        """Computes the Peak Signal-to-Noise Ratio for the given colors.

        Args:
            colors (torch.Tensor): Image colors to compare.

        Returns:
            float: the computed PSNR
        """
        mse = torch.square(colors - self.val_color).mean().item()
        return -10 * math.log10(mse)
