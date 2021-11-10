"""Module providing dataset classes for use in training NeRF models."""

from collections import namedtuple
from enum import Enum
import math
import os
from typing import List, NamedTuple, Union

import cv2
from matplotlib.pyplot import get_cmap
import numpy as np
import scenepic as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .camera_info import CameraInfo
from .utils import download_asset, ETABar
from .raysampler import RaySampler, RaySamples


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
            u = (2 * (row + 0.5) / size) + 1
            for col in range(size):
                v = (2 * (col + 0.5) / size) + 1
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
            u = (2 * (row + 0.5) / size) + 1
            for col in range(size):
                v = (2 * (col + 0.5) / size) + 1
                uvs.append((u, v))

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


class RaySamplesEntry(NamedTuple("RaySamplesEntry", [("samples", RaySamples),
                                                     ("colors", torch.Tensor),
                                                     ("alphas", torch.Tensor)])):
    def to(self, *args) -> "RaySamplesEntry":
        """Calls torch.to on each tensor in the sample."""
        alphas = None if self.alphas is None else self.alphas.to(*args)
        return RaySamplesEntry(self.samples.to(*args),
                               self.colors.to(*args),
                               alphas)

    def pin_memory(self) -> "RaySamplesEntry":
        """Pins all tensors in preparation for movement to the GPU."""
        alphas = None if self.alphas is None else self.alphas.pin_memory()
        return RaySamplesEntry(self.samples.pin_memory(),
                               self.colors.pin_memory(),
                               alphas)

    def subset(self, index: List[int]) -> "RaySamplesEntry":
        """Selects a subset of the samples."""
        alphas = None if self.alphas is None else self.alphas[index]
        return RaySamplesEntry(self.samples.subset(index),
                               self.colors[index],
                               alphas)


class RayDataset(Dataset):
    """Dataset for sampling from rays cast into a volume."""

    class Mode(Enum):
        Full = 0
        Sparse = 1
        Center = 2

    def __init__(self, label: str, images: np.ndarray, bounds: np.ndarray,
                 cameras: List[CameraInfo], num_samples: int,
                 stratified=False, opacity_model: nn.Module = None,
                 batch_size=4096, color_space="RGB"):
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
        assert images.dtype == np.uint8

        self.color_space = color_space
        self.mode = RayDataset.Mode.Full
        self.image_height, self.image_width = images.shape[1:3]
        self.images = images
        self.label = label
        self.sampler = RaySampler(bounds, cameras, num_samples, stratified,
                                  opacity_model, batch_size)

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

        sparse_points = torch.LongTensor(self._subsample_rays(128))
        self.sparse_rays_per_camera = len(sparse_points)

        colors = []
        alphas = []
        crop_index = []
        sparse_index = []
        for image in images:
            color = image[..., :3]
            if color_space == "YCrCb":
                color = cv2.cvtColor(color, cv2.COLOR_RGB2YCrCb)

            color = color.astype(np.float32) / 255
            color = color[self.sampler.points[:, 1],
                          self.sampler.points[:, 0]]
            colors.append(torch.from_numpy(color))

            if image.shape[-1] == 4:
                alpha = image[..., 3].astype(np.float32) / 255
                alpha = alpha[self.sampler.points[:, 1],
                              self.sampler.points[:, 0]]
                alphas.append(torch.from_numpy(alpha))

            offset = len(crop_index) * self.sampler.rays_per_camera
            crop_index.append(crop_points + offset)
            sparse_index.append(sparse_points + offset)

        self.crop_index = torch.cat(crop_index)
        self.sparse_index = torch.cat(sparse_index)

        if len(alphas) > 0:
            self.alphas = torch.cat(alphas)
        else:
            self.alphas = None

        self.colors = torch.cat(colors)

    def to_image(self, colors: np.ndarray) -> np.ndarray:
        pixels = colors.reshape(self.image_height, self.image_width, 3)
        pixels = (pixels * 255).astype(np.uint8)
        if self.color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCB2BGR)
        else:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        return pixels

    @property
    def num_cameras(self) -> bool:
        return self.sampler.num_cameras

    @property
    def num_samples(self) -> int:
        return self.sampler.num_samples

    @property
    def cameras(self) -> List[CameraInfo]:
        return self.sampler.cameras

    def rays_for_camera(self, camera: int) -> RaySamplesEntry:
        """Returns ray samples for the specified camera."""
        if self.mode == RayDataset.Mode.Center:
            start = camera * self.crop_rays_per_camera
            end = start + self.crop_rays_per_camera
        elif self.mode == RayDataset.Mode.Sparse:
            start = camera * self.sparse_rays_per_camera
            end = camera * self.sparse_rays_per_camera
        elif self.mode == RayDataset.Mode.Full:
            start = camera * self.sampler.rays_per_camera
            end = start + self.sampler.rays_per_camera
        else:
            raise NotImplementedError("Unsupported sampling mode")

        return self[list(range(start, end))]

    def __len__(self) -> int:
        """The number of rays in the dataset."""
        if self.mode == RayDataset.Mode.Center:
            return len(self.crop_index)

        if self.mode == RayDataset.Mode.Sparse:
            return len(self.sparse_index)

        if self.mode == RayDataset.Mode.Full:
            return len(self.sampler)

        raise NotImplementedError("Unsupported sampling mode")

    def subset(self, cameras: List[int],
               num_samples: int,
               stratified: bool) -> "RayDataset":
        """Returns a subset of this dataset (by camera).

        Args:
            cameras (List[int]): List of camera indices
            num_samples (int): Number of samples per ray.
            resolution (int): Ray sampling resolution
            stratified (bool): Whether to use stratified sampling.
                               Defaults to False.

        Returns:
            RayDataset: New dataset with the subset of cameras
        """
        return RayDataset(self.label,
                          self.images[cameras],
                          self.sampler.bounds,
                          [self.sampler.cameras[i] for i in cameras],
                          num_samples,
                          stratified,
                          self.sampler.opacity_model,
                          self.sampler.batch_size,
                          self.color_space)

    def sample_cameras(self, num_cameras: int,
                       num_samples: int,
                       stratified: bool) -> "RayDataset":
        """Samples cameras from the dataset and returns the subset.

        Description:
            Cameras are sampled such that they are as equidistant as possible.

        Args:
            num_cameras (int): Number of cameras to sample.
            num_samples (int): Number of samples per ray.
            stratified (bool): Whether to use stratified sampling

        Returns:
            RayDataset: a subset of the dataset with the sampled cameras
        """
        if self.num_cameras < num_cameras:
            samples = list(range(self.num_cameras))
        else:
            positions = np.concatenate([cam.position for cam in self.sampler.cameras])
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

    def __getitem__(self,
                    idx: Union[List[int], torch.Tensor]) -> RaySamplesEntry:
        """Returns samples from the selected rays."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == RayDataset.Mode.Center:
            idx = self.crop_index[idx].tolist()
        elif self.mode == RayDataset.Mode.Sparse:
            idx = self.sparse_index[idx].tolist()

        if not isinstance(idx, list):
            idx = [idx]

        samples = self.sampler[idx]
        colors = self.colors[idx]
        if self.alphas is None:
            alphas = None
        else:
            alphas = self.alphas[idx]

        entry = RaySamplesEntry(samples, colors, alphas)
        entry = entry.pin_memory()
        return entry

    @staticmethod
    def load(path: str, split: str, num_samples: int, stratified: bool,
             opacity_model: nn.Module = None,
             batch_size=4096, color_space="RGB") -> "RayDataset":
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
            RayDataset: A dataset made from the camera and image data
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
        return RayDataset(split, images, bounds, cameras, num_samples,
                          stratified, opacity_model, batch_size,
                          color_space)

    def _subsample_rays(self, resolution: int) -> List[int]:
        num_x_samples = resolution * self.image_width // self.image_height
        num_y_samples = resolution
        x_vals = np.linspace(0, self.image_width - 1, num_x_samples) + 0.5
        y_vals = np.linspace(0, self.image_height - 1, num_y_samples) + 0.5
        x_vals, y_vals = np.meshgrid(x_vals.astype(np.int32),
                                     y_vals.astype(np.int32))
        index = y_vals.reshape(-1) * self.image_width + x_vals.reshape(-1)
        index = index.tolist()
        return index

    def to_scenepic(self, resolution=50) -> sp.Scene:
        """Creates a ray sampling visualization ScenePic for the dataset."""
        scene = sp.Scene()
        frustums = scene.create_mesh("frustums", layer_id="frustums")
        height = 800
        width = height * self.image_width // self.image_height
        canvas = scene.create_canvas_3d(width=width,
                                        height=height)
        canvas.shading = sp.Shading(sp.Colors.Gray)

        idx = np.arange(len(self.sampler.cameras))
        images = self.images
        cameras = self.sampler.cameras

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

        index = self._subsample_rays(resolution)

        bar.finish()

        bar = ETABar("Sampling Rays", max=self.num_cameras)

        bounds = scene.create_mesh("bounds", layer_id="bounds")
        bounds.add_cube(sp.Colors.Blue, transform=self.sampler.bounds)

        frame = canvas.create_frame()
        frame.add_mesh(frustums)
        frame.add_mesh(bounds)
        frame.camera = sp.Camera([0, 0, 10], aspect_ratio=width/height)
        for mesh in image_meshes:
            frame.add_mesh(mesh)

        for i in idx:
            bar.next()
            camera = self.sampler.cameras[i]
            entry = self.rays_for_camera(i)
            entry = entry.subset(index)

            colors = entry.colors.unsqueeze(1)
            colors = colors.expand(-1, self.num_samples, -1)
            positions = entry.samples.positions.numpy().reshape(-1, 3)
            colors = colors.numpy().copy().reshape(-1, 3)

            if entry.alphas is not None:
                alphas = entry.alphas.unsqueeze(1)
                alphas = alphas.expand(-1, self.num_samples)
                alphas = alphas.reshape(-1)
                empty = (alphas < 0.1).numpy()
            else:
                empty = np.zeros_like(colors[..., 0])

            not_empty = np.logical_not(empty)

            samples = scene.create_mesh()
            samples.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.01))
            samples.enable_instancing(positions=positions[not_empty],
                                      colors=colors[not_empty])

            empty_samples = scene.create_mesh(layer_id="empty",
                                              shared_color=sp.Colors.Black)
            empty_samples.add_sphere(transform=sp.Transforms.scale(0.02))
            empty_samples.enable_instancing(positions=positions[empty])

            frame = canvas.create_frame()
            frame.camera = camera.to_scenepic()
            frame.add_mesh(bounds)
            frame.add_mesh(samples)
            frame.add_mesh(empty_samples)
            frame.add_mesh(frustums)
            for mesh in image_meshes:
                frame.add_mesh(mesh)

        canvas.set_layer_settings({
            "bounds": {"opacity": 0.25},
            "images": {"opacity": 0.5}
        })
        bar.finish()

        scene.framerate = 10
        return scene
