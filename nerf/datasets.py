"""Module providing dataset classes for use in training NeRF models."""

from collections import namedtuple
import math
import sys
import time
from typing import List

import cv2
from matplotlib.pyplot import get_cmap
from numba import njit
import numpy as np
import scenepic as sp
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
        mse = torch.square(colors - self.val_color).mean().item()
        return -10 * math.log10(mse)


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
        masks = masks.astype(np.float32)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(1)

        print("Casting rays...")
        x_vals = np.arange(resolution)
        y_vals = np.arange(resolution)
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
        occupancy = F.grid_sample(masks, points, align_corners=True,
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
                       t_starts: np.ndarray,
                       t_ends: np.ndarray,
                       voxel_weights: np.ndarray) -> np.ndarray:
    num_rays, path_length = leaves.shape
    path_length = path_length - 1
    sampling_weights = t_ends - t_starts
    for i in range(num_rays):
        if leaves[i, 0] == -1:
            sampling_weights[i] = 1 / path_length
            continue

        weight_sum = np.sum(sampling_weights[i])
        sampling_weights[i] /= weight_sum

        for j, leaf_index in enumerate(leaves[i]):
            if leaf_index == -1:
                break

            sampling_weights[i, j] = sampling_weights[i, j] * voxel_weights[leaf_index]

    return sampling_weights


@njit
def _sample_t_values(t_starts: np.ndarray, t_ends: np.ndarray,
                     weights: np.ndarray, num_samples: int) -> np.ndarray:
    num_rays = len(t_starts)
    t_values = np.zeros((num_rays, num_samples), np.float32)

    samples = np.random.random(size=(num_rays, num_samples)).astype(np.float32)
    for i in range(num_rays):
        ray_dist = np.cumsum(weights[i])
        ray_dist = ray_dist / ray_dist[-1]
        ray_indices = np.searchsorted(ray_dist, samples[i])
        ray_samples = np.random.random(size=len(ray_indices))
        for j, (index, sample) in enumerate(zip(ray_indices, ray_samples)):
            start = t_starts[i, index]
            end = t_ends[i, index]
            t_values[i, j] = start + sample * (end - start)

        t_values[i] = np.sort(t_values[i])

    return t_values


class RaySamples(namedtuple("RaySample", ["positions", "view_directions",
                                          "deltas", "colors", "alphas",
                                          "t_values"])):
    def to(self, *args) -> "RaySamples":
        return RaySamples(
            self.positions.to(*args),
            self.view_directions.to(*args),
            self.deltas.to(*args),
            self.colors.to(*args),
            self.alphas.to(*args),
            self.t_values.to(*args),
        )


class RaySamplingDataset(Dataset):
    def __init__(self, images: np.ndarray, cameras: List[CameraInfo],
                 num_samples: int, resolution: int, path_length: int = 128,
                 voxels: OcTree = None, voxel_weights: np.ndarray = None,
                 near=2.0, far=6.0, stratified=False):
        """Constructor.

        Args:
            images (np.ndarray): Images of the object from each camera
            cameras (List[CameraInfo]): List of all cameras in the scene
            path_length (int): The maximum number of voxels to intersect with
            num_samples (int): The number of samples to take per ray
            resolution (int): The ray sampling resolution
            voxels (OcTree, optional): The OcTree describing the voxellization
            voxel_weights (np.ndarray, optional): Per-voxel weights to use for
                                                  sampling. Defaults to None
                                                  (i.e. uniform)
            near (float, optional): Near value to use when performing uniform
                                    sampling along rays
            far (float, optional): Far value to use when performing uniform
                                   sampling along rays
            stratified (bool, optional): Whether to use stratified random
                                         sampling
        """
        assert len(images.shape) == 4
        assert len(images) == len(cameras)
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255

        print("Casting rays...")
        source_resolution = images.shape[1]
        scale = source_resolution / resolution

        crop_start = source_resolution // 4
        crop_end = source_resolution - crop_start
        x_vals = ((np.arange(0, resolution) + 0.5) * scale).astype(np.int32)
        y_vals = ((np.arange(0, resolution) + 0.5) * scale).astype(np.int32)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1)
        points = points.reshape(-1, 2)

        inside_crop = (points >= crop_start) & (points < crop_end)
        inside_crop = inside_crop.all(-1)
        crop_points = np.nonzero(inside_crop)[0]

        start = time.time()
        starts = []
        directions = []
        t_starts = []
        t_ends = []
        colors = []
        alphas = []
        weights = []
        crop_index = []
        for camera, image in zip(cameras, images):
            print(camera.name)
            cam_starts, cam_directions = camera.raycast(points)
            cam_colors = image[points[:, 1], points[:, 0]]
            if image.shape[-1] == 4:
                colors.append(cam_colors[..., :3].reshape(-1, 3))
                alphas.append(cam_colors[..., 3].reshape(-1))
            else:
                colors.append(cam_colors.reshape(-1, 3))

            starts.append(cam_starts)
            directions.append(cam_directions)
            crop_index.append(crop_points + len(crop_index) * len(points))
            if voxels is not None:
                path = voxels.intersect(cam_starts, cam_directions, path_length)
                t_starts.append(path.t_stops[:, :-1])
                t_ends.append(path.t_stops[:, 1:])
                weights.append(_determine_weights(path.leaves, t_starts[-1],
                                                  t_ends[-1], voxel_weights))

        starts = np.concatenate(starts)
        directions = np.concatenate(directions)
        colors = np.concatenate(colors)
        crop_index = np.concatenate(crop_index)

        self.starts = torch.from_numpy(starts)
        self.directions = torch.from_numpy(directions)
        self.colors = torch.from_numpy(colors)

        if len(alphas) > 0:
            alphas = np.concatenate(alphas)
            self.alphas = torch.from_numpy(alphas)
        else:
            self.alphas = None

        self.crop_index = torch.from_numpy(crop_index)
        if voxels is None:
            self.uniform_sampling = True
            self.near = near
            self.far = far
            self.stratified = stratified
        else:
            self.t_starts = np.concatenate(t_starts)
            self.t_ends = np.concatenate(t_ends)
            self.weights = np.concatenate(weights)

        passed = time.time() - start
        self.center_crop = False
        self.resolution = resolution
        self.num_rays = len(self.colors)
        self.num_samples = num_samples
        self.images = images
        self.cameras = cameras

        if not self.uniform_sampling:
            print(passed, "elapsed,", self.num_rays, "rays at",
                  passed / self.num_rays, "s/ray")

    def __len__(self) -> int:
        if self.center_crop:
            return len(self.crop_index)

        return self.num_rays

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.center_crop:
            idx = [int(self.crop_index[i]) for i in idx]

        if isinstance(idx, list):
            num_rays = len(idx)
        else:
            num_rays = 1
            idx = [idx]

        starts = self.starts[idx]
        directions = self.directions[idx]

        if self.uniform_sampling:
            t_values = np.linspace(self.near, self.far, self.num_samples,
                                   dtype=np.float32)
            t_values = t_values.reshape(1, self.num_samples)
            if self.stratified:
                scale = (self.far - self.near) / self.num_samples
                permute = np.random.uniform(size=(num_rays, self.num_samples))
                permute = (permute * scale).astype(np.float32)
            else:
                permute = np.zeros((num_rays, self.num_samples), np.float32)

            t_values = t_values + permute
        else:
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
        deltas = deltas.unsqueeze(-1)

        colors = self.colors[idx]
        if len(self.alphas) > 0:
            alphas = self.alphas[idx]
        else:
            alphas = None

        return RaySamples(positions, -directions, deltas,
                          colors, alphas, t_values)

    @staticmethod
    def load(path: str, split: str, resolution: int,
             num_samples: int, stratified: bool) -> "RaySamplingDataset":
        if split == "train":
            idx = list(range(100))
        elif split == "val":
            idx = list(range(100, 107))
        elif split == "test":
            idx = list(range(107, 120))
        else:
            print("Unrecognized split:", split)

        data = np.load(path)
        images = data["images"]
        poses = data["poses"]
        focal_length = data["focal"]
        num_images, height, width = images.shape[:3]
        assert num_images == 120

        cameras = []
        for i, pose in enumerate(poses[idx]):
            name = "{}{:02}".format(split, i)
            intrinsics = np.array([
                [focal_length, 0, width // 2],
                [0, focal_length, height // 2],
                [0, 0, 1]
            ])
            camera_to_world = np.array(pose, np.float32)
            camera_to_world = camera_to_world @ sp.Transforms.rotation_about_x(np.pi)
            cameras.append(CameraInfo.create(name, (width, height),
                                             intrinsics, camera_to_world))

        images = images[idx]
        return RaySamplingDataset(images, cameras, num_samples, resolution,
                                  stratified=stratified)

    def to_scenepic(self) -> sp.Scene:
        scene = sp.Scene()
        frustums = scene.create_mesh("frustums", layer_id="frustums")
        canvas = scene.create_canvas_3d(width=800,
                                        height=800)
        canvas.shading = sp.Shading(sp.Colors.Gray)

        cmap = get_cmap("jet")
        camera_colors = cmap(np.linspace(0, 1, len(self.cameras)))[:, :3]
        image_meshes = []
        sys.stdout.write("Adding cameras")
        for pixels, camera, color in zip(self.images, self.cameras, camera_colors):
            sys.stdout.write(".")
            sys.stdout.flush()
            camera = camera.to_scenepic()

            image = scene.create_image()
            pixels = cv2.resize(pixels, (200, 200), cv2.INTER_AREA)
            image.from_numpy(pixels)
            mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id,
                                     double_sided=True)
            mesh.add_camera_image(camera, depth=0.5)
            image_meshes.append(mesh)

            frustums.add_camera_frustum(camera, color, depth=0.5, thickness=0.01)

        print("done.")

        sys.stdout.write("Sampling rays...")
        num_rays = self.resolution * self.resolution
        for i, camera in enumerate(self.cameras):
            sys.stdout.write(".")
            sys.stdout.flush()
            start = i * num_rays
            end = start + num_rays
            index = [i for i in range(start, end)]
            ray_samples = self[index]

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

        print("done.")

        scene.framerate = 10
        return scene
