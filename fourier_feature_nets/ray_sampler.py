"""Module providing the logic for a ray sampler."""

from typing import List, NamedTuple, Union

import cv2
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

    def numpy(self) -> "RaySamples":
        """Moves all of the tensors from pytorch to numpy."""
        return RaySamples(*[None if tensor is None else tensor.cpu().numpy()
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
        if self.opacity_model is not None:
            self.opacity_model.eval()
            self.focus_sampling = True
        else:
            self.focus_sampling = False

        self.batch_size = batch_size

        x_vals = np.arange(self.image_width)
        y_vals = np.arange(self.image_height)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1)
        self.points = points.reshape(-1, 2)

        num_focus_samples = num_samples - (num_samples // 2)

        self.invalid_rays = set()

        near_far = []
        starts = []
        directions = []
        cdfs = []
        bar = ETABar("Sampling rays", max=len(cameras))
        index = 0
        for camera in cameras:
            bar.next()
            bar.info(camera.name)
            cam_starts, cam_directions = camera.raycast(self.points)
            cam_near_far = self._near_far(cam_starts, cam_directions,
                                          index)
            index += len(self.points)
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

    def to_image(self, camera: int, colors: np.ndarray,
                 color_space: str) -> np.ndarray:
        """Generates an image from the provided ray colors.

        Args:
            camera (int): The camera index, Used to determine a mask for the
                          valid rays in this image.
            colors (np.ndarray): A tensor of pixel colors in the same ray
                                 order as that returned by the sampler.
            color_space (str): The color space used by the colors.

        Returns:
            np.ndarray: A (H,W,3) RGB uint8 image
        """
        idx = self._valid_for_camera(camera)
        idx = [i - camera * self.rays_per_camera for i in idx]
        pixels = np.zeros((self.image_height * self.image_width, 3), np.float32)
        pixels[idx] = colors
        pixels = pixels.reshape(self.image_height, self.image_width, 3)
        pixels = (pixels * 255).astype(np.uint8)
        if color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCB2RGB)

        return pixels

    def _near_far(self, starts: np.ndarray,
                  directions: np.ndarray,
                  index: int) -> np.ndarray:
        # intersect the rays with the bounding volume.
        with np.errstate(divide="ignore", invalid="ignore"):
            # dividing by zero does not hurt us here
            test0 = (self.bounds_min - starts) / directions
            test1 = (self.bounds_max - starts) / directions

        # This test finds the t values at which the ray intersects each
        # axis-aligned plane (i.e. X, Y, Z). We want to have them ordered
        # such that the minimum per-axis t-values are in the near vector,
        # and the maximum ones are in the far vector.
        near = np.where(test0 < test1, test0, test1)
        far = np.where(test0 > test1, test0, test1)

        # This is why the div/0 issue does not affect us. As we only
        # need the max of the mins, and the min of the maxes, we will
        # filter out the NaNs and Infs here.
        near = near.max(-1)
        far = far.min(-1)

        # One nice property here is that if far < near then the
        # ray does not intersect the bounding volume, i.e. it is
        # invalid and won't ever contribute to object pixels. We can
        # ignore it entirely.
        valid = near < far
        near[valid] = np.maximum(0.1, near[valid])
        invalid_index = np.nonzero(np.logical_not(valid))[0] + index
        self.invalid_rays.update(invalid_index.tolist())
        return np.stack([near, far])

    def _determine_opacity(self, t_values: torch.Tensor,
                           starts: torch.Tensor,
                           directions: torch.Tensor) -> torch.Tensor:
        """Use a pretrained opacity model to determine the ray opacity."""
        num_rays = len(starts)
        starts = starts.unsqueeze(1)
        directions = directions.unsqueeze(1)
        t_values = t_values.unsqueeze(2)
        positions = starts + t_values * directions
        device = next(self.opacity_model.parameters()).device
        positions = positions.to(device)
        if self.opacity_model.use_view:
            views = directions.expand(-1, t_values.shape[1], -1)
            views = views.to(device)
        else:
            views = None

        opacity = []
        with torch.no_grad():
            for start in range(0, num_rays, self.batch_size):
                end = min(start + self.batch_size, num_rays)
                batch_pos = positions[start:end]
                batch_pos = batch_pos.reshape(-1, 3)
                batch_view = None
                if self.opacity_model.use_view:
                    batch_view = views[start:end]
                    batch_view = batch_view.reshape(-1, 3)
                    logits = self.opacity_model(batch_pos, batch_view)[:, -1]
                else:
                    logits = self.opacity_model(batch_pos)[:, -1]

                opacity.append(F.softplus(logits))

        opacity = torch.cat(opacity)
        opacity = opacity.reshape(num_rays, -1).cpu()
        return opacity

    def _valid_for_camera(self, camera: int) -> List[int]:
        start = camera * self.rays_per_camera
        end = start + self.rays_per_camera
        idx = list(range(start, end))
        idx = self.to_valid(idx)
        return idx

    def rays_for_camera(self, camera: int) -> RaySamples:
        """Returns the rays for the specified camera."""
        return self[self._valid_for_camera(camera)]

    def to_valid(self, idx: List[int]) -> List[int]:
        """Filters the list of ray indices to include only valid rays.

        Description:
            In this context, a "valid" ray is one which intersects the bounding
            volume.

        Args:
            idx (List[int]): An index of rays in the dataset.

        Returns:
            List[int]: a filtered list of valid rays
        """
        return [i for i in idx if i not in self.invalid_rays]

    def __len__(self) -> int:
        """The number of rays in the dataset."""
        return self.num_rays

    def _sample_t_values(self, idx: List[int], num_samples: int) -> torch.Tensor:
        num_rays = len(idx)

        # first we set up our potential locations
        near, far = self.near_far[:, idx]
        t_values = linspace(near, far, num_samples)
        t_values = 0.5 * (t_values[..., :-1] + t_values[..., 1:])

        if self.stratified:
            samples = torch.rand((num_rays, num_samples), dtype=torch.float32)
        else:
            samples = torch.linspace(0., 1., num_samples)
            samples = samples.unsqueeze(0).repeat(num_rays, 1)

        # get the cumulative distributions for the rays
        cdf = self.cdfs[idx]

        # because it is a cumulative distribution, it is sorted.
        # we can search it to find which t_values should be used
        index = torch.searchsorted(cdf, samples, right=True)

        # now that we know what bin in the CDF each sample falls in, we
        # want to do the following:
        # 1. find the indices below and above the sample position
        # 2. interpolate the t_value for the sample using the CDF
        # this means the final position is a linear interpolation of
        # t = a * t_i + b * t_j where a = (cdf_j - sample)/(cdf_j - cdf_i)
        # and b = (sample - cdf_i)/(cdf_j - cdf_i). We can rewrite this
        # as t = t_i + ((sample - cdf_i) * (t_j - t_i)) / (cdf_j - cdf_i)

        i = torch.maximum(torch.zeros_like(index), index - 1)
        j = torch.minimum(torch.full_like(index, cdf.shape[-1] - 1), index)
        ij = torch.cat([i, j], -1)

        # this gets us cdf_i and cdf_j
        cdf_ij = torch.gather(cdf, 1, ij)
        cdf_ij = cdf_ij.reshape(num_rays, 2, num_samples)

        # this gets us t_i and t_j
        t_ij = torch.gather(t_values, 1, ij)
        t_ij = t_ij.reshape(num_rays, 2, num_samples)

        # (cdf_j - cdf_i)
        denominator = cdf_ij[:, 1] - cdf_ij[:, 0]
        denominator = torch.where(denominator < 1e-5,
                                  torch.ones_like(denominator),
                                  denominator)

        # (sample - cdf_i) / (cdf_j - cdf_i)
        t_diff = (samples - cdf_ij[:, 0]) / denominator

        # (t_j - t_i)
        t_scale = t_ij[:, 1] - t_ij[:, 0]

        samples = t_ij[:, 0] + t_diff * t_scale

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
