"""PyTorch module that models a camera transform from 3D to 2D coordinates."""

from collections import namedtuple
from typing import Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn

from .camera_info import CameraInfo
from .quaternion import qrotate


LANDMARK_SIZE = 32


class CameraTransform(nn.Module):
    """Module modeling the 3D->2D camera transformation.

    Description:
        This module works by transforming 3D coordinates to 2D coordinates via camera projection. While
        The transform object itself can be used as a differentiable camera transform, but can also
        be fit individually via the `fit` method using face landmarks.
    """
    def __init__(self,
                 camera_info: Sequence[CameraInfo]):
        """Initializer.

        Arguments:
            camera_info: the camera information (or a (N,3,3) tensor of camera intrinsics)
        """
        nn.Module.__init__(self)
        projection = torch.from_numpy(np.stack([info.intrinsic for info in camera_info]))

        self._camera_info = camera_info
        center = np.stack([camera.resolution for camera in camera_info])
        center = (center * 0.5).reshape(-1, 1, 2).astype(np.float32)
        center = torch.from_numpy(center)
        self.center = nn.Parameter(center, requires_grad=False)
        self._num_cameras = len(camera_info)
        self.projection = nn.Parameter(projection, requires_grad=False)

        rotation_quat = []
        translate_vec = []
        for cam in camera_info:
            world_to_camera = np.linalg.inv(cam.extrinsic)
            rot = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()
            rotation_quat.append(rot)
            translate_vec.append(world_to_camera[:3, 3])

        rotation_quat = np.stack(rotation_quat).astype(np.float32)
        translate_vec = np.stack(translate_vec).astype(np.float32)

        rotation_quat = torch.from_numpy(rotation_quat)
        translate_vec = torch.from_numpy(translate_vec)

        self.rotation_quat = nn.Parameter(rotation_quat)
        self.translate_vec = nn.Parameter(translate_vec)

    @property
    def camera_info(self) -> Sequence[CameraInfo]:
        """The fitted camera information."""
        return [info.update(np.linalg.inv(world_to_camera), intrinsic)
                for info, world_to_camera, intrinsic in zip(self._camera_info, self.world_to_camera, self.intrinsics)]

    @property
    def world_to_camera(self) -> np.ndarray:
        """The world-to-camera matrices."""
        rquat = self.rotation_quat.detach().cpu().numpy()
        tvec = self.translate_vec.detach().cpu().numpy()
        rotation = Rotation.from_quat(rquat).as_matrix()
        world_to_camera = np.zeros((self._num_cameras, 4, 4), np.float32)
        world_to_camera[:, :3, :3] = rotation
        world_to_camera[:, :3, 3] = tvec
        world_to_camera[:, 3, 3] = 1
        return world_to_camera

    @property
    def intrinsics(self) -> np.ndarray:
        """The intrinsic camera matrices."""
        return self.projection.detach().cpu().numpy()

    def forward(self, positions: torch.Tensor, normalize=False) -> torch.Tensor:
        """Transforms 3D positions into 2D coordinates using a camera projection."""
        num_voxels = positions.shape[0]

        positions = positions.unsqueeze(0)
        positions = positions.expand(self._num_cameras, -1, -1).reshape(-1, 3)

        rquat = self.rotation_quat.unsqueeze(1)
        rquat = rquat.expand(-1, num_voxels, -1).reshape(-1, 4)

        tvec = self.translate_vec.unsqueeze(1)
        tvec = tvec.expand(-1, num_voxels, -1).reshape(-1, 3)

        output = qrotate(rquat, positions)
        output = output + tvec

        projection = self.projection.unsqueeze(1)
        projection = projection.expand(-1, num_voxels, -1, -1).reshape(-1, 3, 3)

        output = torch.bmm(projection, output.unsqueeze(-1)).squeeze(-1)
        output = output[..., :2] / output[..., 2:]
        points = output.reshape(self._num_cameras, num_voxels, 2)

        if normalize:
            points = (points - self.center) / self.center

        return points
