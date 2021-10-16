"""Module providing dataset classes for use in training NeRF models."""

from collections import namedtuple
import math
from typing import List
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from .camera_info import CameraInfo
from .octree import OcTree


class PixelData(namedtuple("PixelData", ["uv", "color"])):
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
        pixels = cv2.imread(path)
        if pixels.shape[0] > pixels.shape[1]:
            start = (pixels.shape[0] - pixels.shape[1]) // 2
            end = start + pixels.shape[1]
            pixels = pixels[start:end, :]
        elif pixels.shape[1] > pixels.shape[0]:
            start = (pixels.shape[1] - pixels.shape[0]) // 2
            end = start + pixels.shape[0]
            pixels = pixels[:, start:end]

        if pixels.shape[0] != size:
            sigma = pixels.shape[0] / size
            pixels = cv2.GaussianBlur(pixels, (0, 0), sigma)
            pixels = cv2.resize(pixels, (size, size), cv2.INTER_NEAREST)

        if color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb) / 255
        elif color_space == "RGB":
            pixels = pixels / 255
        else:
            raise NotImplementedError("Unsupported color space: {}".format(color_space))

        train_uv = []
        train_color = []
        val_uv = []
        val_color = []
        for row in range(size):
            u = (2 * (row + 0.5) / size) - 1
            for col in range(size):
                v = (2 * (col + 0.5) / size) - 1
                color = pixels[row, col].tolist()
                val_uv.append((u, v))
                val_color.append(color)
                if col % 2 or row % 2:
                    train_uv.append((u, v))
                    train_color.append(color)

        train_data = PixelData(torch.FloatTensor(train_uv), torch.FloatTensor(train_color))
        val_data = PixelData(torch.FloatTensor(val_uv), torch.FloatTensor(val_color))
        return PixelDataset(size, color_space, train_data, val_data)

    def to(self, *args) -> "PixelDataset":
        """Equivalent of torch.Tensor.to for all tensors in the dataset.

        Returns:
            PixelDataset: the result of the to operation
        """
        train_data = PixelData(self.train_uv.to(*args), self.train_color.to(*args))
        val_data = PixelData(self.val_uv.to(*args), self.val_color.to(*args))
        return PixelDataset(self.size, self.color_space, train_data, val_data)

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
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCb2BGR)

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
        uvs = []
        for row in range(size):
            u = (2 * (row + 0.5) / size) - 1
            for col in range(size):
                v = (2 * (col + 0.5) / size) - 1
                uvs.append((u, v))

        return torch.FloatTensor(uvs).to(device=device)

    def psnr(self, colors: torch.Tensor) -> float:
        """Computes the Peak Signal-to-Noise Ratio for the given colors.

        Args:
            colors (torch.Tensor): Image colors to compare.

        Returns:
            float: the computed PSNR
        """
        mse = torch.square(255 * (colors - self.val_color)).mean().item()
        return 20 * math.log10(255) - 10 * math.log10(mse)


class VoxelDataset(TensorDataset):
    def __init__(self, masks: np.ndarray, cameras: List[CameraInfo],
                 voxels: OcTree, max_length: int, resolution: int):
        if masks.dtype == np.uint8:
            masks = masks.astype(np.float32) / 255

        self.masks = torch.from_numpy(masks)
        self.masks = self.masks.unsqueeze(1)

        print("Casting rays...")
        x_vals = np.linspace(-1, 1, resolution)
        y_vals = np.linspace(-1, 1, resolution)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1).reshape(1, -1, 2)
        points = points.astype(np.float32)
        start = time.time()
        t_stops = []
        leaves = []
        for camera in cameras:
            print(camera.name)
            starts, directions = camera.raycast(points)
            path = voxels.intersect(starts, directions, max_length)
            t_stops.append(path.t_stops)
            leaves.append(path.leaves)

        points = torch.from_numpy(points)
        points = points.expand(len(cameras), -1, -1)
        points = points.unsqueeze(-2)
        occupancy = F.grid_sample(self.masks, points, align_corners=False,
                                  padding_mode="zeros")
        occupancy = occupancy.reshape(-1)

        t_stops = np.concatenate(t_stops)
        t_stops = torch.from_numpy(t_stops)

        leaves = np.concatenate(leaves)
        leaves = torch.from_numpy(leaves)

        passed = time.time() - start
        num_rays = len(t_stops)
        print(passed, "elapsed,", num_rays, "rays at", passed / num_rays, "s/ray")

        TensorDataset.__init__(self, t_stops, leaves, occupancy)
