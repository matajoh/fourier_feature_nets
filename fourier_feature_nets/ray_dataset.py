"""Module providing a dataset prototype for use in training NeRF models."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Set, Union

import cv2
import numpy as np
import scenepic as sp
import torch

from .camera_info import CameraInfo
from .ray_sampler import RaySamples
from .utils import RenderResult


class RayDataset(ABC):
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

        Patch = 4
        """Returns rays forming distinct patches within the image."""

    @property
    @abstractmethod
    def num_cameras(self) -> int:
        """Number of cameras in the dataset"""

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Number of samples per ray in the dataset"""

    @property
    @abstractmethod
    def color_space(self) -> str:
        """Color space used by the dataset."""

    @property
    @abstractmethod
    def label(self) -> str:
        """A label for the dataset."""

    @property
    @abstractmethod
    def cameras(self) -> List[CameraInfo]:
        """Camera information."""

    @property
    @abstractmethod
    def images(self) -> List[np.ndarray]:
        """Dataset images."""

    @property
    @abstractmethod
    def mode(self) -> "RayDataset.Mode":
        """Sampling mode of the dataset."""

    @mode.setter
    @abstractmethod
    def mode(self, value: "RayDataset.Mode"):
        """Sampling mode of the dataset."""

    @abstractmethod
    def rays_for_camera(self, camera: int) -> RaySamples:
        """Returns ray samples for the specified camera."""

    @property
    @abstractmethod
    def subsample_index(self) -> Set[int]:
        """Set of pixel indices in an image to sample."""

    @subsample_index.setter
    @abstractmethod
    def subsample_index(self, index: Set[int]):
        """Set of pixel indices in an image to sample."""

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
        if len(colors.shape) == 1:
            colors = colors[..., np.newaxis]

        pixels = np.zeros((self.image_height*self.image_width, 3), np.float32)
        index = self.index_for_camera(camera)
        pixels[index] = colors
        pixels = pixels.reshape(self.image_height, self.image_width, 3)
        pixels = (pixels * 255).astype(np.uint8)
        if self._color_space == "YCrCb":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCB2RGB)

        return pixels

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
