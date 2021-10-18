"""Module providing dataset classes for use in training NeRF models."""

from collections import namedtuple
import math
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

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
    """Dataset based on a voxelization of the space."""

    def __init__(self, masks: np.ndarray, cameras: List[CameraInfo],
                 voxels: OcTree, max_length: int, resolution: int):
        """Constructor.

        Args:
            masks (np.ndarray): Binary masks showing the sihouette of an object
                                from each camera.
            cameras (List[CameraInfo]): List of all cameras in the scene
            voxels (OcTree): The OcTree describing the voxellization
            max_length (int): The maximum length of paths through the volume
            resolution (int): The ray sampling resolution
        """
        assert len(masks.shape) == 3
        assert len(masks) == len(cameras)
        masks = np.where(masks > 0, 1, 0)
        if masks.dtype == np.uint8:
            masks = masks.astype(np.float32)

        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(1)

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
        occupancy = F.grid_sample(masks, points, align_corners=False,
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


def _determine_weights(leaves: np.ndarray,
                       voxel_weights: np.ndarray) -> np.ndarray:
    num_rays, path_length = leaves.shape
    sampling_weights = np.zeros((num_rays, path_length), np.float32)
    for i in range(num_rays):
        if leaves[i, 0] == -1:
            sampling_weights[i] = 1 / path_length
            continue

        for j, leaf_index in enumerate(leaves[i]):
            if leaf_index == -1:
                break
           
            sampling_weights[i, j] = voxel_weights[leaf_index]

    return sampling_weights


def _sample_t_values(t_starts: np.ndarray, t_ends: np.ndarray,
                     weights: np.ndarray, num_samples: int) -> np.ndarray:
    num_rays = len(t_starts)
    t_values = np.zeros((num_rays, num_samples), np.float32)
    weights = np.cumsum(weights, -1)
    weights = weights / weights[:, -1:]

    samples = np.random.random(size=(num_rays, num_samples)).astype(np.float32)
    for i in range(num_rays):
        indices = np.searchsorted(weights[i], samples[i])
        t_values[i] = np.random.uniform(t_starts[i, indices],
                                        t_ends[i, indices])

    t_values = np.sort(t_values, -1)
    return t_values


class RaySamplingDataset(Dataset):
    def __init__(self, images: np.ndarray, cameras: List[CameraInfo],
                 voxels: OcTree, path_length: int, num_samples: int,
                 resolution: int, voxel_weights: np.ndarray = None):
        """Constructor.

        Args:
            images (np.ndarray): Images of the object from each camera
            cameras (List[CameraInfo]): List of all cameras in the scene
            voxels (OcTree): The OcTree describing the voxellization
            path_length (int): The maximum number of voxels to intersect with
            num_samples (int): The number of samples to take per ray
            resolution (int): The ray sampling resolution
            voxel_weights (np.ndarray, optional): Per-voxel weights to use for
                                                  sampling. Defaults to None
                                                  (i.e. uniform)
        """
        assert len(images.shape) == 4
        assert len(images) == len(cameras)
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255

        images = images.transpose(0, 3, 1, 2)
        images = torch.from_numpy(images)

        if voxel_weights is None:
            voxel_weights = np.ones(voxels.num_leaves, np.float32)

        print("Casting rays...")
        x_vals = np.linspace(-1, 1, resolution)
        y_vals = np.linspace(-1, 1, resolution)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1).reshape(1, -1, 2)
        points = points.astype(np.float32)
        start = time.time()
        starts = []
        directions = []
        t_stops = []
        weights = []
        for camera in cameras:
            print(camera.name)
            cam_starts, cam_directions = camera.raycast(points)
            starts.append(cam_starts)
            directions.append(cam_directions)
            path = voxels.intersect(cam_starts, cam_directions, path_length)
            t_stops.append(path.t_stops)
            weights.append(_determine_weights(path.leaves, voxel_weights))

        points = torch.from_numpy(points)
        points = points.expand(len(cameras), -1, -1)
        points = points.unsqueeze(-2)
        colors = F.grid_sample(images, points, align_corners=False,
                               padding_mode="zeros")
        colors = colors.transpose(1, 2)
        self.colors = colors.reshape(-1, 3)

        starts = np.concatenate(starts)
        directions = np.concatenate(directions)
        t_stops = np.concatenate(t_stops)
        weights = np.concatenate(weights)

        self.starts = torch.from_numpy(starts)
        self.directions = torch.from_numpy(directions)
        self.t_starts = t_stops[:, :-1]
        self.t_ends = t_stops[:, 1:]
        self.weights = weights[:, :-1]

        passed = time.time() - start
        self.num_rays = len(t_stops)
        self.num_samples = num_samples
        print(passed, "elapsed,", self.num_rays, "rays at",
              passed / self.num_rays, "s/ray")

    def __len__(self) -> int:
        return self.num_rays

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        num_rays = len(idx)
        starts = self.starts[idx]
        directions = self.directions[idx]

        t_starts = self.t_starts[idx]
        t_ends = self.t_ends[idx]
        weights = self.weights[idx]

        t_values = _sample_t_values(t_starts, t_ends, weights, self.num_samples)
        starts = starts.reshape(num_rays, 1, 3)
        directions = directions.reshape(num_rays, 1, 3)
        t_values = torch.from_numpy(t_values)
        positions = starts + t_values.unsqueeze(-1) * directions

        max_dist = torch.full((num_rays, 1), 1e10, dtype=torch.float32)
        deltas = t_values[:, 1:] - t_values[:, :-1]
        deltas = torch.cat([deltas, max_dist], axis=-1)

        colors = self.colors[idx]

        return positions, deltas, colors
