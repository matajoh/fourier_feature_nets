"""Module providing dataset classes for use in training NeRF models."""

from enum import Enum
import os
from typing import List, NamedTuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scenepic as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .camera_info import CameraInfo, Resolution
from .ray_sampler import RaySampler, RaySamples
from .utils import download_asset, ETABar


class RayData(NamedTuple("RayData", [("samples", RaySamples),
                                     ("colors", torch.Tensor),
                                     ("alphas", torch.Tensor)])):
    """Class representing ray data, with samples, colors and alpha values."""

    def to(self, *args) -> "RayData":
        """Calls torch.to on each tensor in the sample."""
        alphas = None if self.alphas is None else self.alphas.to(*args)
        return RayData(self.samples.to(*args),
                       self.colors.to(*args),
                       alphas)

    def pin_memory(self) -> "RayData":
        """Pins all tensors in preparation for movement to the GPU."""
        alphas = None if self.alphas is None else self.alphas.pin_memory()
        return RayData(self.samples.pin_memory(),
                       self.colors.pin_memory(),
                       alphas)

    def subset(self, index: List[int]) -> "RayData":
        """Selects a subset of the samples."""
        alphas = None if self.alphas is None else self.alphas[index]
        return RayData(self.samples.subset(index),
                       self.colors[index],
                       alphas)

    def numpy(self) -> "RayData":
        """Moves the tensors from pytorch to numpy."""
        alphas = None if self.alphas is None else self.alphas.cpu().numpy()
        return RayData(self.samples.numpy(),
                       self.colors.cpu().numpy(),
                       alphas)


class RayDataset(Dataset):
    """Dataset for sampling from rays cast into a volume."""

    class Mode(Enum):
        """The sampling mode of the dataset."""
        Full = 0
        """All valid rays are returned."""

        Sparse = 1
        """Returns a subsampled version of each image."""

        Center = 2
        """Returns a center crop of the image."""

        Dilate = 3
        """Returns rays from a dilated region around the alpha mask."""

    def __init__(self, label: str, images: np.ndarray, bounds: np.ndarray,
                 cameras: List[CameraInfo], num_samples: int,
                 include_alpha=True, stratified=False,
                 opacity_model: nn.Module = None,
                 batch_size=4096, color_space="RGB",
                 sparse_size=50):
        """Constructor.

        Args:
            label (str): Label used to identify this dataset.
            images (np.ndarray): Images of the object from each camera
            bounds (np.ndarray): Bounds of the render volume defined as a
                                 transform matrix on the unit cube.
            cameras (List[CameraInfo]): List of all cameras in the scene
            num_samples (int): The number of samples to take per ray
            include_alpha (bool): Whether to include alpha if present
            stratified (bool, optional): Whether to use stratified random
                                         sampling
            opacity_model (nn.Module, optional): Optional model which predicts
                                                 opacity in the volume, used
                                                 for performing targeted
                                                 sampling if provided. Defaults
                                                 to None.
            batch_size (int, optional): Batch size to use with the opacity
                                        model. Defaults to 4096.
            color_space (str, optional): The color space to use. Defaults to
                                         "RGB".
            sparse_size (int, optional): The vertical resolution of
                                         the sparse sampling version.
        """
        assert len(images.shape) == 4
        assert len(images) == len(cameras)
        assert images.dtype == np.uint8

        self.color_space = color_space
        self._mode = RayDataset.Mode.Full
        self.image_height, self.image_width = images.shape[1:3]
        self.images = images
        self.label = label
        self.include_alpha = include_alpha
        self.subsample_index = None
        self.sampler = RaySampler(bounds, cameras, num_samples, stratified,
                                  opacity_model, batch_size)

        source_resolution = np.array([self.image_width, self.image_height],
                                     np.float32)
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

        sparse_points = torch.LongTensor(self._subsample_rays(sparse_size))
        sparse_height = sparse_size
        sparse_width = sparse_size * self.image_width // self.image_height
        self.sparse_resolution = sparse_width, sparse_height
        self.sparse_rays_per_camera = len(sparse_points)

        stencil_radius = 8 * min(self.image_width, self.image_height) // 100
        size = 2 * stencil_radius + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        self.dilate_ranges = []
        num_dilate = 0

        colors = []
        alphas = []
        crop_index = []
        sparse_index = []
        dilate_index = []
        bar = ETABar("Indexing", max=len(images))
        for image in images:
            bar.next()
            color = image[..., :3]
            if color_space == "YCrCb":
                color = cv2.cvtColor(color, cv2.COLOR_RGB2YCrCb)

            color = color.astype(np.float32) / 255
            color = color[self.sampler.points[:, 1],
                          self.sampler.points[:, 0]]
            colors.append(torch.from_numpy(color))

            offset = len(crop_index) * self.sampler.rays_per_camera
            if image.shape[-1] == 4:
                alpha = image[..., 3].astype(np.float32) / 255
                mask = (alpha > 0).astype(np.uint8)

                alpha = alpha[self.sampler.points[:, 1],
                              self.sampler.points[:, 0]]
                alphas.append(torch.from_numpy(alpha))

                mask = cv2.dilate(mask, element)
                mask = mask[self.sampler.points[:, 1],
                            self.sampler.points[:, 0]]
                dilate_points, = np.nonzero(mask)
                dilate_index.append(torch.from_numpy(dilate_points) + offset)
                start = num_dilate
                end = start + len(dilate_points)
                num_dilate = end
                self.dilate_ranges.append((start, end))

            crop_index.append(crop_points + offset)
            sparse_index.append(sparse_points + offset)

        bar.finish()
        self.crop_index = torch.cat(crop_index)
        self.sparse_index = torch.cat(sparse_index)
        self.dilate_index = torch.cat(dilate_index)

        if len(alphas) > 0 and include_alpha:
            self.alphas = torch.cat(alphas)
        else:
            self.alphas = None

        self.colors = torch.cat(colors)

    @property
    def resolution(self) -> Resolution:
        """The resolution of the images."""
        return Resolution(self.image_width, self.image_height)

    @property
    def mode(self) -> "RayDataset.Mode":
        """The current sampling mode of the dataset."""
        return self._mode

    @mode.setter
    def mode(self, value: "RayDataset.Mode"):
        if value == RayDataset.Mode.Dilate and len(self.dilate_index) == 0:
            raise ValueError("Unable to use dilate mode: missing alpha channel")

        self._mode = value

    def to_image(self, camera: int, colors: np.ndarray) -> np.ndarray:
        """Creates an image given the camera and the compute pixel colors.

        Args:
            camera (int): The camera index. Needed to handle special patterns,
                          i.e. for Dilate mode.
            colors (np.ndarray): The computed colors, one per ray, in the order
                                 returned by the dataset.

        Returns:
            np.ndarray: A (H,W,3) uint8 RGB tensor
        """
        pixels = np.zeros((self.image_height*self.image_width, 3), np.float32)
        index = self.index_for_camera(camera)
        pixels[index] = colors
        pixels = pixels.reshape(self.image_height, self.image_width, 3)
        pixels = (pixels * 255).astype(np.uint8)
        if self.color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCB2RGB)

        return pixels

    @property
    def num_cameras(self) -> bool:
        """Number of cameras in the dataset."""
        return self.sampler.num_cameras

    @property
    def num_samples(self) -> int:
        """Number of samples per ray."""
        return self.sampler.num_samples

    @property
    def cameras(self) -> List[CameraInfo]:
        """Camera information."""
        return self.sampler.cameras

    def index_for_camera(self, camera: int) -> List[int]:
        """Returns a pixel index for the camera.

        Description:
            This method will take into account special patterns from sampling,
            such as sparsity, center cropping, or dilation.

        Args:
            camera (int): the camera index

        Returns:
            List[int]: index into the rays for this camera
        """
        camera_start = camera * self.sampler.rays_per_camera
        if self.mode == RayDataset.Mode.Center:
            start = camera * self.crop_rays_per_camera
            end = start + self.crop_rays_per_camera
            idx = self.crop_index[start:end].tolist()
        elif self.mode == RayDataset.Mode.Sparse:
            start = camera * self.sparse_rays_per_camera
            end = start + self.sparse_rays_per_camera
            idx = self.sparse_index[start:end].tolist()
        elif self.mode == RayDataset.Mode.Dilate:
            start, end = self.dilate_ranges[camera]
            idx = self.dilate_index[start:end].tolist()
        elif self.mode == RayDataset.Mode.Full:
            camera_end = camera_start + self.sampler.rays_per_camera
            idx = list(range(camera_start, camera_end))
        else:
            raise NotImplementedError("Unsupported sampling mode")

        idx = self.sampler.to_valid(idx)
        idx = [i - camera_start for i in idx]
        return idx

    def rays_for_camera(self, camera: int) -> RayData:
        """Returns ray samples for the specified camera."""
        if self.mode == RayDataset.Mode.Center:
            start = camera * self.crop_rays_per_camera
            end = start + self.crop_rays_per_camera
        elif self.mode == RayDataset.Mode.Sparse:
            start = camera * self.sparse_rays_per_camera
            end = start + self.sparse_rays_per_camera
        elif self.mode == RayDataset.Mode.Dilate:
            start, end = self.dilate_ranges[camera]
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

        if self.mode == RayDataset.Mode.Dilate:
            return len(self.dilate_index)

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
                          self.include_alpha,
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
                    idx: Union[List[int], torch.Tensor]) -> RayData:
        """Returns samples from the selected rays."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == RayDataset.Mode.Center:
            idx = self.crop_index[idx].tolist()
        elif self.mode == RayDataset.Mode.Sparse:
            idx = self.sparse_index[idx].tolist()
        elif self.mode == RayDataset.Mode.Dilate:
            idx = self.dilate_index[idx].tolist()

        if not isinstance(idx, list):
            idx = [idx]

        if self.subsample_index:
            idx = [i for i in idx
                   if i % self.sampler.rays_per_camera in self.subsample_index]

        idx = self.sampler.to_valid(idx)
        samples = self.sampler[idx]
        colors = self.colors[idx]
        if self.alphas is None or self.mode == RayDataset.Mode.Dilate:
            alphas = None
        else:
            alphas = self.alphas[idx]
            colors = torch.where(alphas.unsqueeze(1) > 0, colors,
                                 torch.zeros_like(colors))

        entry = RayData(samples, colors, alphas)
        entry = entry.pin_memory()
        return entry

    @staticmethod
    def load(path: str, split: str, num_samples: int,
             include_alpha: bool, stratified: bool,
             opacity_model: nn.Module = None,
             batch_size=4096, color_space="RGB",
             sparse_size=50) -> "RayDataset":
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
            include_alpha (bool): Whether to include alpha if present
            stratified (bool): whether to use stratified sampling.
            opacity_model (nn.Module, optional): model that predicts opacity
                                                 from 3D position. If the model
                                                 predicts more than one value,
                                                 the last channel is used.
                                                 Defaults to None.
            batch_size (int, optional): Batch size to use when sampling the
                                        opacity model.
            sparse_size (int, optional): Resolution for sparse sampling.

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
                                     Resolution(width, height),
                                     intr, extr)
                   for i, (intr, extr) in enumerate(zip(intrinsics,
                                                        extrinsics))]
        return RayDataset(split, images, bounds, cameras, num_samples,
                          include_alpha, stratified, opacity_model,
                          batch_size, color_space, sparse_size)

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

    def to_scenepic(self) -> sp.Scene:
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

        cmap = plt.get_cmap("jet")
        camera_colors = cmap(np.linspace(0, 1, len(cameras)))[:, :3]
        image_meshes = []
        bar = ETABar("Plotting cameras", max=self.num_cameras)
        thumb_height = 200
        thumb_width = thumb_height * self.image_width // self.image_height
        for i, pixels, camera, color in zip(idx, images,
                                            cameras, camera_colors):
            bar.next()
            camera = camera.to_scenepic()

            image = scene.create_image()
            cam_index = self.index_for_camera(i)
            pixels = (pixels / 255).astype(np.float32)
            pixels = pixels[..., :3].reshape(-1, 3)[cam_index]
            pixels = self.to_image(i, pixels)
            pixels = cv2.resize(pixels, (thumb_width, thumb_height),
                                cv2.INTER_AREA)
            image.from_numpy(pixels)
            mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id,
                                     double_sided=True)
            mesh.add_camera_image(camera, depth=0.5)
            image_meshes.append(mesh)

            frustums.add_camera_frustum(camera, color, depth=0.5, thickness=0.01)

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

        sampling_mode = self.mode
        for cam in idx:
            bar.next()
            camera = self.sampler.cameras[cam]

            self.mode = RayDataset.Mode.Sparse
            index = set(self.index_for_camera(cam))
            self.mode = sampling_mode
            index.intersection_update(self.index_for_camera(cam))
            self.mode = RayDataset.Mode.Full
            cam_start = cam * self.sampler.rays_per_camera
            index = [cam_start + i for i in index]
            entry = self[index]

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

            samples = scene.create_mesh(layer_id="samples")
            samples.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.01))
            samples.enable_instancing(positions=positions[not_empty],
                                      colors=colors[not_empty])

            frame = canvas.create_frame()

            if empty.any():
                empty_samples = scene.create_mesh(layer_id="empty samples")
                empty_samples.add_sphere(sp.Colors.Black,
                                         transform=sp.Transforms.scale(0.01))
                empty_samples.enable_instancing(positions=positions[empty],
                                                colors=colors[empty])
                frame.add_mesh(empty_samples)

            frame.camera = camera.to_scenepic()
            frame.add_mesh(bounds)
            frame.add_mesh(samples)
            frame.add_mesh(frustums)
            for mesh in image_meshes:
                frame.add_mesh(mesh)

        self.mode = sampling_mode

        canvas.set_layer_settings({
            "bounds": {"opacity": 0.25},
            "images": {"opacity": 0.5}
        })
        bar.finish()

        scene.framerate = 10
        return scene
