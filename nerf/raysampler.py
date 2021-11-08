from typing import List, NamedTuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_info import CameraInfo
from .utils import calculate_blend_weights, ETABar, linspace


class RaySamples(NamedTuple("RaySamples", [("positions", torch.Tensor),
                                           ("view_directions", torch.Tensor),
                                           ("t_values", torch.Tensor)])):
    """Points samples from rays.

    Description:
        Each sample is the result of the following equation:

            start + direction * t

        Where start is the position of the camera, and direction is the
        direction of a ray passing through a pixel in an image from that
        camera. The samples consist of the following parts:

            positions: the 3D positions
            view_directions: the direction from each position back to the camera
            t_values: the t_values corresponding to the positions

        Each tensor is grouped by ray, so the first two dimensions will be
        (num_rays,num_samples).
    """
    def to(self, *args) -> "RaySamples":
        """Calls torch.to on each tensor in the sample."""
        return RaySamples(*[None if tensor is None else tensor.to(*args)
                            for tensor in self])

    def pin_memory(self) -> "RaySamples":
        """Pins all tensors in preparation for movement to the GPU."""
        return RaySamples(*[None if tensor is None else tensor.pin_memory()
                            for tensor in self])

    def subset(self, index: List[int]) -> "RaySamples":
        """Selects a subset of the samples."""
        return RaySamples(*[None if tensor is None else tensor[index]
                            for tensor in self])


def _determine_cdf(t_values: torch.Tensor,
                   opacity: torch.Tensor) -> torch.Tensor:
    weights = calculate_blend_weights(t_values, opacity)
    weights = weights[:, 1:-1]
    weights += 1e-5
    cdf = weights.cumsum(-1)
    cdf = cdf / cdf[:, -1:]
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    return cdf


class RaySampler:
    """Dataset for sampling from rays cast into a volume."""

    def __init__(self, bounds: np.ndarray,
                 cameras: List[CameraInfo], num_samples: int,
                 stratified=False, opacity_model: nn.Module = None,
                 batch_size=4096):
        """Constructor.

        Args:
            bounds (np.ndarray): Bounds of the render volume defined as a
                                 transform matrix on the unit cube.
            cameras (List[CameraInfo]): List of all cameras in the scene
            num_samples (int): The number of samples to take per ray
            stratified (bool, optional): Whether to use stratified random
                                         sampling
            opacity_model (nn.Module, optional): Optional model which predicts
                                                 opacity in the volume, used
                                                 for performing targeted
                                                 sampling if provided. Defaults
                                                 to None.
            batch_size (int, optional): Batch size to use with the opacity
                                        model. Defaults to 4096.
        """
        self.bounds = bounds
        bounds_min = bounds @ np.array([-0.5, -0.5, -0.5, 1], np.float32)
        bounds_max = bounds @ np.array([0.5, 0.5, 0.5, 1], np.float32)
        self.bounds_min = bounds_min[np.newaxis, :3]
        self.bounds_max = bounds_max[np.newaxis, :3]
        self.image_width, self.image_height = cameras[0].resolution
        self.rays_per_camera = self.image_width * self.image_height
        self.num_rays = len(cameras) * self.rays_per_camera
        self.num_cameras = len(cameras)
        self.num_samples = num_samples
        print({
            "width": self.image_width,
            "height": self.image_height,
            "rays_per_camera": self.rays_per_camera,
            "num_cameras": self.num_cameras,
            "num_rays": self.num_rays,
            "num_samples": self.num_samples
        })
        self.cameras = cameras
        self.stratified = stratified
        self.opacity_model = opacity_model
        self.focus_sampling = opacity_model is not None
        self.batch_size = batch_size

        x_vals = np.arange(self.image_width)
        y_vals = np.arange(self.image_height)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1)
        self.points = points.reshape(-1, 2)

        num_focus_samples = num_samples - (num_samples // 2)

        near_far = []
        starts = []
        directions = []
        cdfs = []
        bar = ETABar("Sampling rays", max=len(cameras))
        for camera in cameras:
            bar.next()
            bar.info(camera.name)
            cam_starts, cam_directions = camera.raycast(self.points)
            cam_near_far = self._near_far(cam_starts, cam_directions)
            cam_starts = torch.from_numpy(cam_starts)
            cam_directions = torch.from_numpy(cam_directions)
            cam_near_far = torch.from_numpy(cam_near_far)
            starts.append(cam_starts)
            directions.append(cam_directions)
            near_far.append(cam_near_far)
            if self.focus_sampling:
                t_values = linspace(cam_near_far[0], cam_near_far[1],
                                    num_focus_samples)
                opacity = self._determine_opacity(t_values, cam_starts,
                                                  cam_directions)
                cdfs.append(_determine_cdf(t_values, opacity))

        self.starts = torch.cat(starts)
        self.directions = torch.cat(directions)
        self.near_far = torch.cat(near_far, -1)

        if self.focus_sampling:
            self.cdfs = torch.cat(cdfs)

        bar.finish()

    def _near_far(self, starts: np.ndarray,
                  directions: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            test0 = (self.bounds_min - starts) / directions
            test1 = (self.bounds_max - starts) / directions

        near = np.where(test0 < test1, test0, test1)
        far = np.where(test0 > test1, test0, test1)

        near = near.max(-1)
        far = far.min(-1)

        valid = near < far
        near[:] = np.maximum(0.1, near[valid].min())
        far[:] = far[valid].max()
        return np.stack([near, far])

    def _determine_opacity(self, t_values: torch.Tensor,
                           starts: torch.Tensor,
                           directions: torch.Tensor) -> torch.Tensor:
        num_rays = len(starts)
        starts = starts.unsqueeze(1)
        directions = directions.unsqueeze(1)
        t_values = t_values.unsqueeze(2)
        positions = starts + t_values * directions
        device = next(self.opacity_model.parameters()).device
        positions = positions.to(device)
        opacity = []
        with torch.no_grad():
            for start in range(0, num_rays, self.batch_size):
                end = min(start + self.batch_size, num_rays)
                batch = positions[start:end]
                batch = batch.reshape(-1, 3)
                logits = self.opacity_model(batch)[:, -1]
                opacity.append(F.softplus(logits))

        opacity = torch.cat(opacity)
        opacity = opacity.reshape(num_rays, -1).cpu()
        return opacity

    def rays_for_camera(self, camera: int) -> RaySamples:
        """Returns the rays for the specified camera."""
        start = camera * self.rays_per_camera
        end = start + self.rays_per_camera
        return self[list(range(start, end))]

    def __len__(self) -> int:
        """The number of rays in the dataset."""
        return self.num_rays

    def _sample_t_values(self, idx: List[int], num_samples: int) -> torch.Tensor:
        num_rays = len(idx)

        near, far = self.near_far[:, idx]
        t_values = linspace(near, far, num_samples)
        t_values = 0.5 * (t_values[..., :-1] + t_values[..., 1:])

        if self.stratified:
            samples = torch.rand((num_rays, num_samples), dtype=torch.float32)
        else:
            samples = torch.linspace(0., 1., num_samples)
            samples = samples.unsqueeze(0).repeat(num_rays, 1)

        cdf = self.cdfs[idx]
        index = torch.searchsorted(cdf, samples, right=True)
        index_below = torch.maximum(torch.zeros_like(index), index - 1)
        index_above = torch.minimum(torch.full_like(index, cdf.shape[-1] - 1),
                                    index)
        index_ba = torch.cat([index_below, index_above], -1)

        cdf_ba = torch.gather(cdf, 1, index_ba)
        cdf_ba = cdf_ba.reshape(num_rays, 2, num_samples)
        t_values_ba = torch.gather(t_values, 1, index_ba)
        t_values_ba = t_values_ba.reshape(num_rays, 2, num_samples)

        denominator = cdf_ba[:, 1] - cdf_ba[:, 0]
        denominator = torch.where(denominator < 1e-5,
                                  torch.ones_like(denominator),
                                  denominator)
        t_diff = (samples - cdf_ba[:, 0]) / denominator
        t_scale = t_values_ba[:, 1] - t_values_ba[:, 0]
        samples = t_values_ba[:, 0] + t_diff * t_scale

        return samples

    def __getitem__(self, idx: Union[List[int], torch.Tensor]) -> RaySamples:
        """Returns the requested sampled rays."""
        num_rays = len(idx)

        starts = self.starts[idx]
        directions = self.directions[idx]

        if self.focus_sampling:
            num_samples = self.num_samples // 2
        else:
            num_samples = self.num_samples

        near, far = self.near_far[:, idx]
        t_values = linspace(near, far, num_samples)
        if self.stratified:
            scale = (far - near) / num_samples
            permute = torch.rand((num_rays, num_samples),
                                 dtype=torch.float32)
            permute = permute * scale.unsqueeze(-1)
            t_values = t_values + permute

        if self.focus_sampling:
            num_focus_samples = self.num_samples - num_samples
            focus_t_values = self._sample_t_values(idx, num_focus_samples)
            t_values = torch.cat([t_values, focus_t_values], -1)
            t_values, _ = t_values.sort(-1)

        starts = starts.reshape(num_rays, 1, 3)
        directions = directions.reshape(num_rays, 1, 3)
        directions = directions.repeat(1, self.num_samples, 1)
        positions = starts + t_values.unsqueeze(-1) * directions

        ray_samples = RaySamples(positions, directions, t_values)
        return ray_samples
