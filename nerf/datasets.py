"""Module providing dataset classes for use in training NeRF models."""

from collections import namedtuple
import math
import os
from typing import List, NamedTuple, Union

import cv2
from matplotlib.pyplot import get_cmap
import numpy as np
import scenepic as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .camera_info import CameraInfo
from .utils import calculate_blend_weights, download_asset, ETABar


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
            sigma = 0.5 * pixels.shape[0] / size
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
            u = (row + 0.5) / size
            for col in range(size):
                v = (col + 0.5) / size
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
        vals = np.linspace(0, 1, size)
        uvs = np.stack(np.meshgrid(vals, vals, indexing="ij"), -1)
        return torch.FloatTensor(uvs).to(device=device)

    def psnr(self, colors: torch.Tensor) -> float:
        """Computes the Peak Signal-to-Noise Ratio for the given colors.

        Args:
            colors (torch.Tensor): Image colors to compare.

        Returns:
            float: the computed PSNR
        """
        mse = torch.square(colors - self.val_color).mean().item()
        return -10 * math.log10(mse)


def _determine_cdf(t_values: torch.Tensor,
                   opacity: torch.Tensor) -> torch.Tensor:
    weights = calculate_blend_weights(t_values, opacity)
    weights = weights[:, 1:-1]
    weights += 1e-5
    cdf = weights.cumsum(-1)
    cdf = cdf / cdf[:, -1:]
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    return cdf


class RaySamples(NamedTuple("RaySamples", [("positions", torch.Tensor),
                                           ("view_directions", torch.Tensor),
                                           ("colors", torch.Tensor),
                                           ("alphas", torch.Tensor),
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
            colors: the pixel colors for the ray
            alphas: the alpha values for the ray (if present in the data)
            t_values: the t_values corresponding to the positions

        Each tensor is grouped by ray, so the first two dimensions will be
        (num_rays,num_samples).
    """
    def to(self, *args) -> "RaySamples":
        """Calls torch.to on each tensor in the sample."""
        return RaySamples(*[tensor.to(*args) for tensor in self])

    def pin_memory(self) -> "RaySamples":
        """Pins all tensors in preparation for movement to the GPU."""
        return RaySamples(*[tensor.pin_memory() for tensor in self])

    def subset(self, index: List[int]) -> "RaySamples":
        """Selects a subset of the samples."""
        return RaySamples(*[tensor[index] for tensor in self])


def _linspace(start: torch.Tensor, stop: torch.Tensor, num_samples: int):
    diff = stop - start
    samples = torch.linspace(0, 1, num_samples)
    return start.unsqueeze(-1) + samples.unsqueeze(0) * diff.unsqueeze(-1)


class RaySamplingDataset(Dataset):
    """Dataset for sampling from rays cast into a volume."""

    def __init__(self, label: str, images: np.ndarray, bounds: np.ndarray,
                 cameras: List[CameraInfo], num_samples: int,
                 stratified=False, opacity_model: nn.Module = None,
                 batch_size=4096):
        """Constructor.

        Args:
            label (str): Label used to identify this dataset.
            images (np.ndarray): Images of the object from each camera
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
        assert len(images.shape) == 4
        assert len(images) == len(cameras)
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255

        self.bounds = bounds
        bounds_min = bounds @ np.array([-0.5, -0.5, -0.5, 1], np.float32)
        bounds_max = bounds @ np.array([0.5, 0.5, 0.5, 1], np.float32)
        self.bounds_min = bounds_min[np.newaxis, :3]
        self.bounds_max = bounds_max[np.newaxis, :3]
        self.label = label
        self.center_crop = False
        self.image_height, self.image_width = images.shape[1:3]
        self.rays_per_camera = self.image_width * self.image_height
        self.num_rays = len(cameras) * self.rays_per_camera
        self.num_cameras = len(cameras)
        self.num_samples = num_samples
        self.images = images
        self.cameras = cameras
        self.stratified = stratified
        self.opacity_model = opacity_model
        self.focus_sampling = opacity_model is not None
        self.batch_size = batch_size

        source_resolution = np.array(images.shape[1:3])

        crop_start = source_resolution // 4
        crop_end = source_resolution - crop_start
        x_vals = np.arange(self.image_width)
        y_vals = np.arange(self.image_height)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1)
        points = points.reshape(-1, 2)

        inside_crop = (points >= crop_start) & (points < crop_end)
        inside_crop = inside_crop.all(-1)
        crop_points = np.nonzero(inside_crop)[0]
        crop_points = torch.from_numpy(crop_points)
        self.crop_rays_per_camera = len(crop_points)

        num_focus_samples = num_samples - (num_samples // 2)

        near_far = []
        starts = []
        directions = []
        colors = []
        alphas = []
        crop_index = []
        cdfs = []
        bar = ETABar("Loading {} data".format(label), max=len(cameras))
        for camera, image in zip(cameras, images):
            bar.next()
            bar.info(camera.name)
            cam_starts, cam_directions = camera.raycast(points)
            cam_near_far = self._near_far(cam_starts, cam_directions)
            cam_starts = torch.from_numpy(cam_starts)
            cam_directions = torch.from_numpy(cam_directions)
            cam_near_far = torch.from_numpy(cam_near_far)
            cam_colors = image[points[:, 1], points[:, 0]]
            cam_colors = torch.from_numpy(cam_colors)
            if image.shape[-1] == 4:
                colors.append(cam_colors[..., :3].reshape(-1, 3))
                alphas.append(cam_colors[..., 3].reshape(-1))
            else:
                colors.append(cam_colors.reshape(-1, 3))

            starts.append(cam_starts)
            directions.append(cam_directions)
            near_far.append(cam_near_far)
            crop_index.append(crop_points + len(crop_index) * len(points))
            if self.focus_sampling:
                t_values = _linspace(cam_near_far[0], cam_near_far[1],
                                     num_focus_samples)
                opacity = self._determine_opacity(t_values, cam_starts,
                                                  cam_directions)
                cdfs.append(_determine_cdf(t_values, opacity))

        self.crop_index = torch.cat(crop_index)
        self.starts = torch.cat(starts)
        self.directions = torch.cat(directions)
        self.near_far = torch.cat(near_far)

        if len(alphas) > 0:
            self.alphas = torch.cat(alphas)
        else:
            self.alphas = None

        self.colors = torch.cat(colors)

        if self.focus_sampling:
            self.cdfs = torch.cat(cdfs)

        bar.finish()

    def _near_far(self, starts: np.ndarray,
                  directions: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            test0 = (self.bounds_min - starts) / directions
            test1 = (self.bounds_max - starts) / directions

        int0 = np.where(test0 < test1, test0, test1)
        int1 = np.where(test0 > test1, test0, test1)

        int0 = int0.max(-1)
        int1 = int1.min(-1)
        near = np.where(int0 < int1, int0, int1)
        far = np.where(int0 > int1, int0, int1)
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
        if self.center_crop:
            start = camera * self.crop_rays_per_camera
            end = start + self.crop_rays_per_camera
        else:
            start = camera * self.rays_per_camera
            end = start + self.rays_per_camera

        return self[list(range(start, end))]

    def __len__(self) -> int:
        """The number of rays in the dataset."""
        if self.center_crop:
            return len(self.crop_index)

        return self.num_rays

    def subset(self, cameras: List[int],
               num_samples: int,
               stratified: bool) -> "RaySamplingDataset":
        """Returns a subset of this dataset (by camera).

        Args:
            cameras (List[int]): List of camera indices
            num_samples (int): Number of samples per ray.
            resolution (int): Ray sampling resolution
            stratified (bool): Whether to use stratified sampling.
                               Defaults to False.

        Returns:
            RaySamplingDataset: New dataset with the subset of cameras
        """
        return RaySamplingDataset(self.label,
                                  self.images[cameras],
                                  self.bounds,
                                  [self.cameras[i] for i in cameras],
                                  num_samples,
                                  stratified,
                                  self.opacity_model,
                                  self.batch_size)

    def sample_cameras(self, num_cameras: int,
                       num_samples: int,
                       stratified: bool) -> "RaySamplingDataset":
        """Samples cameras from the dataset and returns the subset.

        Description:
            Cameras are sampled such that they are as equidistant as possible.

        Args:
            num_cameras (int): Number of cameras to sample.
            num_samples (int): Number of samples per ray.
            stratified (bool): Whether to use stratified sampling

        Returns:
            RaySamplingDataset: a subset of the dataset with the sampled cameras
        """
        if self.num_cameras < num_cameras:
            samples = list(range(self.num_cameras))
        else:
            positions = np.concatenate([cam.position for cam in self.cameras])
            samples = set([0])
            all_directions = set(range(len(positions)))
            while len(samples) < num_cameras:
                sample_positions = positions[list(samples)]
                distances = positions[:, None, :] - sample_positions[None, :, :]
                distances = np.square(distances).sum(-1).min(-1)
                unchosen = np.array(list(all_directions - samples))
                distances = np.array(distances[unchosen], np.float32)
                choice = unchosen[distances.argmax()]
                samples.add(choice)

        return self.subset(list(samples), num_samples, stratified)

    def _sample_t_values(self, idx: List[int], num_samples: int) -> torch.Tensor:
        num_rays = len(idx)

        t_values = _linspace(self.near_far[0], self.near_far[1], num_samples)
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.center_crop:
            idx = self.crop_index[idx].tolist()

        if isinstance(idx, list):
            num_rays = len(idx)
        else:
            num_rays = 1
            idx = [idx]

        starts = self.starts[idx]
        directions = self.directions[idx]

        if self.focus_sampling:
            num_samples = self.num_samples // 2
        else:
            num_samples = self.num_samples

        t_values = _linspace(self.near_far[0], self.near_far[1], num_samples)
        if self.stratified:
            scale = (self.near_far[1] - self.near_far[0]) / num_samples
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

        colors = self.colors[idx]
        if len(self.alphas) > 0:
            alphas = self.alphas[idx]
        else:
            alphas = None

        ray_samples = RaySamples(positions, directions, colors,
                                 alphas, t_values)
        ray_samples = ray_samples.pin_memory()
        return ray_samples

    @staticmethod
    def load(path: str, split: str, num_samples: int, stratified: bool,
             opacity_model: nn.Module = None,
             batch_size=4096) -> "RaySamplingDataset":
        """Loads a dataset from an NPZ file.

        Description:
            The NPZ file should contain the following elements:

            images: a (NxRxRx[3,4]) tensor of images in RGB(A) format.
            bounds: a (4x4) transform from the unit cube to a render volume
            intrinsics: a (Nx3x3) tensor of camera intrinsics (projection)
            extrinsics: a (Nx4x4) tensor of camera extrinsics (camera to world)
            split_counts: a (3) tensor of counts per split in train, val, test
                          order

        Args:
            path (str): path to an NPZ file containing the dataset
            split (str): the split to load [train, val, test]
            num_samples (int): the number of samples per ray
            stratified (bool): whether to use stratified sampling.
            opacity_model (nn.Module, optional): model that predicts opacity
                                                 from 3D position. If the model
                                                 predicts more than one value,
                                                 the last channel is used.
                                                 Defaults to None.
            batch_size (int, optional): Batch size to use when sampling the
                                        opacity model.

        Returns:
            RaySamplingDataset: A dataset made from the camera and image data
        """
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(__file__), "..", "data", path)
            path = os.path.abspath(path)
            if not os.path.exists(path):
                print("Downloading dataset...")
                dataset_name = os.path.basename(path)
                success = download_asset(dataset_name, path)
                if not success:
                    print("Unable to download dataset", dataset_name)
                    return None

        data = np.load(path)
        test_end, height, width = data["images"].shape[:3]
        split_counts = data["split_counts"]
        train_end = split_counts[0]
        val_end = train_end + split_counts[1]

        if split == "train":
            idx = list(range(train_end))
        elif split == "val":
            idx = list(range(train_end, val_end))
        elif split == "test":
            idx = list(range(val_end, test_end))
        else:
            print("Unrecognized split:", split)
            return None

        bounds = data["bounds"]
        images = data["images"][idx]
        intrinsics = data["intrinsics"][idx]
        extrinsics = data["extrinsics"][idx]

        cameras = [CameraInfo.create("{}{:03}".format(split, i),
                                     (width, height),
                                     intr, extr)
                   for i, (intr, extr) in enumerate(zip(intrinsics,
                                                        extrinsics))]
        return RaySamplingDataset(split, images, bounds, cameras, num_samples,
                                  stratified, opacity_model, batch_size)

    def to_scenepic(self, resolution=50) -> sp.Scene:
        """Creates a ray sampling visualization ScenePic for the dataset."""
        scene = sp.Scene()
        frustums = scene.create_mesh("frustums", layer_id="frustums")
        height = 800
        width = height * self.image_width // self.image_height
        canvas = scene.create_canvas_3d(width=width,
                                        height=height)
        canvas.shading = sp.Shading(sp.Colors.Gray)

        idx = np.arange(len(self.cameras))
        images = self.images
        cameras = self.cameras

        cmap = get_cmap("jet")
        camera_colors = cmap(np.linspace(0, 1, len(cameras)))[:, :3]
        image_meshes = []
        bar = ETABar("Plotting cameras", max=self.num_cameras)
        for pixels, camera, color in zip(images, cameras, camera_colors):
            bar.next()
            camera = camera.to_scenepic()

            image = scene.create_image()
            pixels = cv2.resize(pixels, (200, 200), cv2.INTER_AREA)
            image.from_numpy(pixels[..., :3])
            mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id,
                                     double_sided=True)
            mesh.add_camera_image(camera, depth=0.5)
            image_meshes.append(mesh)

            frustums.add_camera_frustum(camera, color, depth=0.5, thickness=0.01)

        num_x_samples = resolution * self.image_width // self.image_height
        num_y_samples = resolution
        x_vals = np.linspace(0, self.image_width - 1, num_x_samples) + 0.5
        y_vals = np.linspace(0, self.image_height - 1, num_y_samples) + 0.5
        x_vals, y_vals = np.meshgrid(x_vals.astype(np.int32),
                                     y_vals.astype(np.int32))
        index = y_vals.reshape(-1) * self.image_width + x_vals.reshape(-1)
        index = index.tolist()

        bar.finish()

        bar = ETABar("Sampling Rays", max=self.num_cameras)

        frame = canvas.create_frame()
        frame.add_mesh(frustums)
        for mesh in image_meshes:
            frame.add_mesh(mesh)

        for i in idx:
            bar.next()
            camera = self.cameras[i]
            ray_samples = self.rays_for_camera(i)
            ray_samples = ray_samples.subset(index)

            colors = ray_samples.colors.unsqueeze(1)
            colors = colors.expand(-1, self.num_samples, -1)
            positions = ray_samples.positions.numpy().reshape(-1, 3)
            colors = colors.numpy().copy().reshape(-1, 3)

            empty = (colors == 0).sum(-1) == 3
            not_empty = np.logical_not(empty)

            samples = scene.create_mesh()
            samples.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.02))
            samples.enable_instancing(positions=positions[not_empty],
                                      colors=colors[not_empty])

            empty_samples = scene.create_mesh(layer_id="empty",
                                              shared_color=sp.Colors.Black)
            empty_samples.add_sphere(transform=sp.Transforms.scale(0.02))
            empty_samples.enable_instancing(positions=positions[empty])

            frame = canvas.create_frame()
            frame.camera = camera.to_scenepic()
            frame.add_mesh(samples)
            frame.add_mesh(empty_samples)
            frame.add_mesh(frustums)
            for mesh in image_meshes:
                frame.add_mesh(mesh)

        bar.finish()

        scene.framerate = 10
        return scene
