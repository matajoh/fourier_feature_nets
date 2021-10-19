from collections import namedtuple
import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .camera_info import CameraInfo
from .datasets import VoxelDataset
from .octree import OcTree


class VolumeCarver(nn.Module):
    def __init__(self, depth: int, scale: float, path_length: int,
                 image_dir: str, blob_prior_weight: float, resolution: int):
        nn.Module.__init__(self)
        self.voxels = OcTree(depth, scale)
        print(depth, len(self.voxels))
        self._path_length = path_length
        self._image_dir = image_dir
        self._blob_prior_weight = blob_prior_weight
        self._resolution = resolution
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        logits = torch.full((self.voxels.num_leaves,), -10.0, dtype=torch.float32)
        self.logits = nn.Parameter(logits)

        self._blob_prior = nn.Parameter(self._compute_distance_weights())

        self.log = []

    def _compute_distance_weights(self) -> torch.Tensor:
        positions = torch.from_numpy(self.voxels.leaf_centers())
        squared_dist = positions.square().sum(-1)
        squared_dist /= self.voxels.scale * self.voxels.scale * 3
        return squared_dist

    def forward(self, t_stops: torch.Tensor,
                leaves: torch.Tensor) -> torch.Tensor:
        num_rays, length = leaves.shape
        max_dist = torch.full((num_rays, 1), 1e10, dtype=torch.float32,
                              device=self.logits.device)
        left_trans = torch.ones_like(max_dist)
        opacity = torch.sigmoid(self.logits)
        opacity = opacity[leaves]
        deltas = t_stops[:, 1:] - t_stops[:, :-1]
        deltas = torch.cat([deltas, max_dist], axis=-1)
        alpha = 1 - torch.exp(-(opacity * deltas))
        ones = torch.ones_like(alpha)
        trans = torch.minimum(ones, 1 - alpha + 1e-10)
        _, trans = trans.split([1, length - 1], dim=-1)
        trans = torch.cat([left_trans, trans], -1)
        weights = alpha * torch.cumprod(trans, -1)
        output = weights.sum(-1)
        return output

    def _val_image(self, camera: CameraInfo) -> np.ndarray:
        x_vals = np.linspace(-1, 1, self._resolution)
        y_vals = np.linspace(-1, 1, self._resolution)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1).reshape(-1, 2)
        starts, directions = camera.raycast(points)
        t_stops, leaves = self.voxels.intersect(starts, directions,
                                                self._path_length)
        t_stops = torch.from_numpy(t_stops)
        leaves = torch.from_numpy(leaves)
        pixels = self.forward(t_stops, leaves)
        pixels = pixels.detach().cpu().numpy()
        pixels = pixels.reshape(self._resolution, self._resolution)
        pixels = (pixels * 255).astype(np.uint8)
        return pixels

    def _loss(self, t_stops: torch.Tensor, leaves: torch.Tensor,
              targets: torch.Tensor, debug: bool) -> torch.Tensor:
        outputs = self.forward(t_stops, leaves)
        render_energy = (targets - outputs).square().mean()
        opacity = torch.sigmoid(self.logits)
        blob_energy = (self._blob_prior * opacity).square().sum()
        loss = render_energy + self._blob_prior_weight * blob_energy
        if debug:
            return namedtuple("Loss", ["render", "blob"])(
                render_energy.item(),
                blob_energy.item()
            )

        return loss

    @property
    def opacity(self) -> np.ndarray:
        return torch.sigmoid(self.logits).detach().cpu().numpy()

    def _log(self, iteration, loss):
        occupied = (self.opacity * 5).astype(np.int32)
        print(iteration, loss,
              np.bincount(occupied, minlength=5).tolist())

    def fit(self, images: torch.Tensor, cameras: List[CameraInfo],
            batch_size: int, num_epochs: int, learning_rate=1.0):
        optim = torch.optim.Adam([self.logits], learning_rate)

        dataset = VoxelDataset(images, cameras, self.voxels,
                               self._path_length, self._resolution)
        data_loader = DataLoader(dataset, batch_size, True)

        device = self.logits.device
        viz_camera = 0
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            for step, (t_stops, leaves, targets) in enumerate(data_loader):
                t_stops = t_stops.to(device)
                leaves = leaves.to(device)
                targets = targets.to(device)

                optim.zero_grad()
                loss = self._loss(t_stops, leaves, targets, False)
                loss.backward()
                optim.step()

                if step % 5 == 0:
                    with torch.no_grad():
                        self._log(step, self._loss(t_stops, leaves, targets, True))

                    with torch.no_grad():
                        name = "e{:04}_s{:03}_c{:03}.png".format(epoch, step, viz_camera)
                        path = os.path.join(self._image_dir, name)
                        cv2.imwrite(path, self._val_image(cameras[viz_camera]))
                        viz_camera = (viz_camera + 1) % len(cameras)
