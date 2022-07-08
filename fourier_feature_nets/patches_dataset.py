"""Module providing dataset classes for use in training NeRF models."""

from typing import List, Set, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scenepic as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .camera_info import CameraInfo
from .ray_dataset import RayDataset
from .ray_sampler import RaySampler, RaySamples
from .utils import ETABar, RenderResult


class PatchesDataset(Dataset, RayDataset):
    def __init__(self, label: str, bounds: np.ndarray,
                 cameras: List[CameraInfo], num_samples: int,
                 num_patches: int, patch_size: int,
                 stratified=False, opacity_model: nn.Module = None,
                 batch_size=4096, color_space="RGB",
                 anneal_start=0.2, num_anneal_steps=0,
                 depth_weight_start=400.0, depth_weight_end=10,
                 num_depth_steps=512):
        """Constructor.

        Args:
            label (str): Label used to identify this dataset.
            bounds (np.ndarray): Bounds of the render volume defined as a
                                 transform matrix on the unit cube.
            cameras (List[CameraInfo]): List of all cameras in the scene
            num_samples (int): The number of samples to take per ray
            num_patches (int): The number of patches per camera
            patch_size (int): The size of the square patch
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
            anneal_start (float, optiona): Starting value for the sample space
                                           annealing. Defaults to 0.2.
            num_anneal_steps (int, optional): Steps over which to anneal
                                              sampling to the full range of
                                              volume intersection. Defaults
                                              to 0.
            depth_weight_start (float, optional): Starting depth reg weight.
                                                  Defaults to 400.
            depth_weight_end (float, optional): Ending depth reg weight.
                                                Defaults to 10.
            num_depth_steps (int, optional): Number of steps over which to
                                             anneal the depth weight.
        """
        self._label = label
        self._num_samples = num_samples
        self._cameras = cameras
        self._color_space = color_space
        self._subsample_index = None
        self._mode = RayDataset.Mode.Patch
        self._sampler = RaySampler(bounds, cameras, num_samples, stratified,
                                   opacity_model, batch_size, anneal_start,
                                   num_anneal_steps)

        self._depth_weight_start = depth_weight_start
        self._depth_weight_end = depth_weight_end
        self._num_depth_steps = num_depth_steps

        grid_rows = int(np.sqrt(num_patches))
        grid_cols = num_patches // grid_rows
        if grid_rows * grid_cols < num_patches:
            grid_cols += 1

        self.patches_per_camera = num_patches
        self.patch_size = patch_size
        self.rays_per_patch = patch_size * patch_size

        self.sparse_rays_per_camera = num_patches

        patches_index = []
        self.patches_ranges = []
        sparse_index = []
        bar = ETABar("Indexing", max=len(cameras))
        patch_start = 0
        for i, camera in enumerate(cameras):
            bar.next()

            resolution = camera.resolution
            patch_points = []
            for r in range(patch_size):
                for c in range(patch_size):
                    patch_points.append(r * resolution.width + c)

            sparse_points = [(patch_size * resolution.width + patch_size) // 2]

            patch_points = torch.LongTensor(patch_points)
            sparse_points = torch.LongTensor(sparse_points)

            cell_height = (resolution.height - patch_size) // grid_rows
            cell_width = (resolution.width - patch_size) // grid_cols

            x_perturb = max(cell_width - patch_size, patch_size // 2)
            y_perturb = max(cell_height - patch_size, patch_size // 2)

            x, y = np.meshgrid(np.arange(grid_cols), np.arange(grid_rows))
            x = (x * cell_width).reshape(-1)
            y = (y * cell_height).reshape(-1)

            x += np.random.random_integers(0, x_perturb, x.shape)
            y += np.random.random_integers(0, y_perturb, y.shape)

            index = np.arange(len(x))
            np.random.shuffle(index)
            index = index[:num_patches]
            offset = i * self._sampler.rays_per_camera
            patches = []
            for r, c in zip(y[index], x[index]):
                patch_offset = r * resolution.width + c
                points = patch_points + offset + patch_offset
                valid_points = self._sampler.to_valid(points.tolist())
                if len(valid_points) < len(points):
                    continue

                patches.append(points)
                sparse_index.append(sparse_points + offset + patch_offset)
                if len(patches) == num_patches:
                    break

            patch_end = patch_start + len(patches)
            self.patches_ranges.append((patch_start, patch_end))
            patch_start = patch_end
            patches = torch.stack(patches)
            patches_index.append(patches)

        bar.finish()
        self.patches_index = torch.cat(patches_index)
        self.sparse_index = torch.cat(sparse_index)

    @property
    def num_cameras(self) -> int:
        """Number of cameras in the dataset"""
        return len(self._cameras)

    @property
    def num_samples(self) -> int:
        """Number of samples per ray in the dataset"""
        return self._num_samples

    @property
    def color_space(self) -> str:
        """Color space used by the dataset."""
        return self._color_space

    @property
    def label(self) -> str:
        """A label for the dataset."""
        return self._label

    @property
    def cameras(self) -> List[CameraInfo]:
        """Camera information."""
        return self._cameras

    @property
    def images(self) -> List[np.ndarray]:
        """Dataset images."""
        raise NotImplementedError("not image based")

    @property
    def mode(self) -> RayDataset.Mode:
        """Sampling mode of the dataset."""
        return self._mode

    @mode.setter
    def mode(self, mode: RayDataset.Mode):
        """Sampling mode of the dataset."""
        self._mode = mode

    def rays_for_camera(self, camera: int) -> RaySamples:
        """Returns ray samples for the specified camera."""
        if self.mode == RayDataset.Mode.Patch:
            start, end = self.patches_ranges[camera]
        elif self.mode == RayDataset.Mode.Sparse:
            start = camera * self.sparse_rays_per_camera
            end = start + self.sparse_rays_per_camera
        elif self.mode == RayDataset.Mode.Full:
            start = camera * self._sampler.rays_per_camera
            end = start + self._sampler.rays_per_camera
        else:
            raise NotImplementedError("Unsupported sampling mode")

        return self.get_rays(list(range(start, end)), None)

    @property
    def subsample_index(self) -> Set[int]:
        """Set of pixel indices in an image to sample."""
        return self._subsample_index

    @subsample_index.setter
    def subsample_index(self, index: Set[int]):
        """Set of pixel indices in an image to sample."""
        self._subsample_index = index

    def loss(self, step: int, _: RaySamples, render: RenderResult) -> torch.Tensor:
        """Compute the dataset loss for the prediction.

        Args:
            actual (RaySamples): The rays to render
            predicted (RenderResult): The ray rendering result

        Returns:
            torch.Tensor: a scalar loss tensor
        """
        depth = render.depth.reshape(-1, self.patch_size, self.patch_size)
        d00 = depth[:, :-1, :-1]
        d01 = depth[:, :-1, 1:]
        d10 = depth[:, 1:, :-1]

        loss = torch.square(d00 - d01) + torch.square(d00 - d10)

        if self._num_depth_steps:
            progress = np.clip(step / self._num_depth_steps, 0, 1)
            weight = self._depth_weight_start * (1 - progress)
            weight = weight + progress * self._depth_weight_end
        else:
            weight = self._depth_weight_end

        loss = loss.mean() * weight
        assert not loss.isnan()
        return loss

    def get_rays(self,
                 idx: Union[List[int], torch.Tensor],
                 step: int = None) -> RaySamples:
        """Returns samples from the selected rays.

        Args:
            idx (Union[List[int], torch.Tensor]): index into the dataset
            step (int, optional): Step in optimization. Defaults to None.

        Returns:
            (RaySamples): Returns ray data
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == RayDataset.Mode.Patch:
            idx = self.patches_index[idx]
            idx = idx.reshape(-1).tolist()
        elif self.mode == RayDataset.Mode.Sparse:
            idx = self.sparse_index[idx].tolist()

        if not isinstance(idx, list):
            idx = [idx]

        if self.subsample_index:
            idx = [i for i in idx
                   if i % self._sampler.rays_per_camera in self.subsample_index]

        return self._sampler.sample(idx, step)

    def render(self, rays: RaySamples) -> RenderResult:
        """Returns a (ground truth) render of the rays.

        Args:
            rays (RaySamples): the rays to render

        Returns:
            RenderResult: the ground truth render
        """
        raise NotImplementedError("not supported")

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
        camera_start = camera * self._sampler.rays_per_camera
        if self.mode == RayDataset.Mode.Patch:
            start, end = self.patches_ranges[camera]
            idx = self.patches_index[start:end]
            idx = idx.reshape(-1).tolist()
        elif self.mode == RayDataset.Mode.Sparse:
            start = camera * self.sparse_rays_per_camera
            end = start + self.sparse_rays_per_camera
            idx = self.sparse_index[start:end].tolist()
        elif self.mode == RayDataset.Mode.Full:
            camera_end = camera_start + self._sampler.rays_per_camera
            idx = list(range(camera_start, camera_end))
        else:
            raise NotImplementedError("Unsupported sampling mode")

        idx = [i - camera_start for i in idx]
        return idx

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
        return self._sampler.to_valid(idx)

    def __len__(self) -> int:
        """The number of rays in the dataset."""
        if self.mode == RayDataset.Mode.Patch:
            return len(self.patches_index)

        if self.mode == RayDataset.Mode.Sparse:
            return len(self.sparse_index)

        if self.mode == RayDataset.Mode.Full:
            return len(self._sampler)

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
        return PatchesDataset(self.label,
                              self._sampler.bounds,
                              [self._sampler.cameras[i] for i in cameras],
                              num_samples,
                              self.patches_per_camera,
                              self.patch_size,
                              stratified,
                              self._sampler.opacity_model,
                              self._sampler.batch_size,
                              self.color_space,
                              self._sampler.anneal_start,
                              self._sampler.num_anneal_steps)

    def to_scenepic(self) -> sp.Scene:
        """Creates a ray sampling visualization ScenePic for the dataset."""
        scene = sp.Scene()
        frustums = scene.create_mesh("frustums", layer_id="frustums")
        resolution = self.cameras[0].resolution
        canvas_res = resolution.scale_to_height(800)
        canvas = scene.create_canvas_3d(width=canvas_res.width,
                                        height=canvas_res.height)
        canvas.shading = sp.Shading(sp.Colors.Gray)

        idx = np.arange(len(self._sampler.cameras))
        cameras = self._sampler.cameras

        cmap = plt.get_cmap("jet")
        camera_colors = cmap(np.linspace(0, 1, len(cameras)))[:, :3]
        image_meshes = []
        bar = ETABar("Plotting cameras", max=self.num_cameras)
        thumb_res = resolution.scale_to_height(200)

        sparse_colors = patch_colors = np.arange(self.patches_per_camera)
        patch_colors = patch_colors.repeat(self.patch_size * self.patch_size)
        patch_colors = cmap(patch_colors / (self.patches_per_camera - 1))
        patch_colors = patch_colors[..., :3]
        patch_colors = (patch_colors * 255).astype(np.uint8)
        sparse_colors = cmap(sparse_colors / (self.patches_per_camera - 1))
        sparse_colors = sparse_colors[..., :3].astype(np.float32)
        for i, camera, color in zip(idx, cameras, camera_colors):
            bar.next()
            width, height = camera.resolution
            camera = camera.to_scenepic()

            image = scene.create_image()
            pixels = np.zeros((width * height, 3), np.uint8)
            start = i * self.patches_per_camera
            end = start + self.patches_per_camera
            index = self.patches_index[start:end].reshape(-1)
            index = index - i * self._sampler.rays_per_camera
            index = index.tolist()
            pixels[index] = patch_colors
            pixels = pixels.reshape(height, width, 3)
            pixels = cv2.resize(pixels, thumb_res, cv2.INTER_AREA)
            image.from_numpy(pixels)
            mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id,
                                     double_sided=True)
            mesh.add_camera_image(camera, depth=0.5)
            image_meshes.append(mesh)

            frustums.add_camera_frustum(camera, color, depth=0.5, thickness=0.01)

        bar.finish()

        bar = ETABar("Sampling Rays", max=self.num_cameras)

        bounds = scene.create_mesh("bounds", layer_id="bounds")
        bounds.add_cube(sp.Colors.Blue, transform=self._sampler.bounds)

        frame = canvas.create_frame()
        frame.add_mesh(frustums)
        frame.add_mesh(bounds)
        frame.camera = sp.Camera([0, 0, 10], aspect_ratio=resolution.ratio)
        for mesh in image_meshes:
            frame.add_mesh(mesh)

        sampling_mode = self.mode
        for cam in idx:
            bar.next()
            camera = self._sampler.cameras[cam]

            self.mode = RayDataset.Mode.Sparse
            index = set(self.index_for_camera(cam))
            self.mode = sampling_mode
            index.intersection_update(self.index_for_camera(cam))
            self.mode = RayDataset.Mode.Full
            cam_start = cam * self._sampler.rays_per_camera
            index = [cam_start + i for i in index]
            samples = self.get_rays(index)
            colors = sparse_colors[:len(index)]
            colors = colors[:, np.newaxis]
            colors = np.repeat(colors, self.num_samples, 1)
            colors = colors.reshape(-1, 3)

            positions = samples.positions.numpy().reshape(-1, 3)

            samples = scene.create_mesh(layer_id="samples")
            samples.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.01))
            samples.enable_instancing(positions=positions,
                                      colors=colors)

            frame = canvas.create_frame()

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
