"""Module providing dataset classes for use in training NeRF models."""

from typing import List, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .camera_info import CameraInfo
from .ray_dataset import RayDataset
from .ray_sampler import RaySampler, RaySamples
from .utils import ETABar, RenderResult


class RegularizerDataset(Dataset, RayDataset):
    def __init__(self, label: str, bounds: np.ndarray,
                 cameras: List[CameraInfo], num_samples: int, patch_size: int,
                 stratified=False, opacity_model: nn.Module = None,
                 batch_size=4096, color_space="RGB",
                 anneal_start=0.2, num_anneal_steps=0):
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
        """
        self._label = label
        self._num_samples = num_samples
        self._cameras = cameras
        self._color_space = color_space
        self._subsample_index = None
        self._sampler = RaySampler(bounds, cameras, num_samples, stratified,
                                   opacity_model, batch_size, anneal_start,
                                   num_anneal_steps)

        colors = []
        alphas = []
        patch_index = []
        bar = ETABar("Indexing", max=len(cameras))
        for camera in cameras:
            bar.next()

            # select patches distributed evenly
            # perturb the positions
            
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
        return RayDataset.Mode.Patch

    @mode.setter
    def mode(self, _: RayDataset.Mode):
        """Sampling mode of the dataset."""
        raise NotImplementedError("only patch mode is supported")

    def rays_for_camera(self, camera: int) -> RaySamples:
        """Returns ray samples for the specified camera."""
        if self.mode == RayDataset.Mode.Patch:
            start = camera * self.patch_rays_per_camera
            end = start + self.patch_rays_per_camera
        elif self.mode == RayDataset.Mode.Full:
            start = camera * self.sampler.rays_per_camera
            end = start + self.sampler.rays_per_camera
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

    @abstractmethod
    def loss(self, rays: RaySamples, render: RenderResult) -> torch.Tensor:
        """Compute the dataset loss for the prediction.

        Args:
            actual (RaySamples): The rays to render
            predicted (RenderResult): The ray rendering result

        Returns:
            torch.Tensor: a scalar loss tensor
        """

    @abstractmethod
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

    @abstractmethod
    def render(self, rays: RaySamples) -> RenderResult:
        """Returns a (ground truth) render of the rays.

        Args:
            rays (RaySamples): the rays to render

        Returns:
            RenderResult: the ground truth render
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def __len__(self) -> int:
        """The number of rays in the dataset."""

    @abstractmethod
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

    @abstractmethod
    def to_scenepic(self) -> sp.Scene:
        """Creates a ray sampling visualization ScenePic for the dataset."""
