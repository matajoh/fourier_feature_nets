"""Module defining various visualizers that can be used during training."""

from abc import ABC, abstractmethod
import os
from typing import Callable

import cv2
import numpy as np

from .camera_info import Resolution
from .image_dataset import ImageDataset
from .ray_sampler import RaySampler, RaySamples
from .utils import orbit, RenderResult

ImageRender = Callable[[RaySamples, bool], RenderResult]
ActivationRender = Callable[[RaySampler, int], np.ndarray]


class Visualizer(ABC):
    """A visualizer can hook into the training process to produce artifacts."""
    @abstractmethod
    def visualize(self, step: int, render: ImageRender,
                  act_render: ActivationRender):
        """Create a visualization using the provided render functions.

        Args:
            step (int): Step in the optimization
            render (ImageRender): Render function in image space
            act_render (ActivationRender): Render function for the activations
        """


class EvaluationVisualizer(Visualizer):
    """Produces image grids showing GT, prediction, depth, and error."""

    def __init__(self, results_dir: str, dataset: ImageDataset, interval: int,
                 max_depth=10):
        """Constructor.

        Args:
            results_dir (str): the base results directory.
            dataset (ImageDataset): the dataset to use as reference.
            interval (int): the number of steps between images.
            max_depth (int, optional): Value used to clip the depth.
                                       Defaults to 10.
        """
        path = os.path.join(results_dir, dataset.label)
        os.makedirs(path, exist_ok=True)
        self._output_dir = path
        self._dataset = dataset
        self._interval = interval
        self._index = 0
        self._max_depth = max_depth

    def visualize(self, step: int, render: ImageRender,
                  _: ActivationRender):
        """Create a visualization using the provided render functions.

        Args:
            step (int): Step in the optimization
            render (ImageRender): Render function in image space
            act_render (ActivationRender): Render function for the activations
        """
        if step % self._interval != 0:
            return

        camera = self._index % self._dataset.num_cameras
        samples = self._dataset.rays_for_camera(camera)
        act = self._dataset.render(samples).numpy()
        pred = render(samples, True)

        error = np.square(act.color - pred.color).sum(-1)
        if act.alpha is not None:
            error = 3 * error
            error += np.square(act.alpha - pred.alpha)
            error = error / 4

        width, height = self._dataset.cameras[camera].resolution
        predicted = np.clip(pred.color, 0, 1)
        predicted_image = self._dataset.to_image(camera, predicted)

        actual_image = self._dataset.to_image(camera, act.color)

        depth = np.clip(pred.depth, 0, self._max_depth) / self._max_depth
        depth_image = self._dataset.to_image(camera, depth)

        error = np.sqrt(error)
        error = error / error.max()
        error_image = self._dataset.to_image(camera, error)

        name = "s{:07}_c{:03}.png".format(step, camera)
        image_path = os.path.join(self._output_dir, name)

        compare = np.zeros((height*2, width*2, 3), np.uint8)
        compare[:height, :width] = predicted_image
        compare[height:, :width] = actual_image
        compare[:height, width:] = depth_image
        compare[height:, width:] = error_image
        compare = cv2.cvtColor(compare, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, compare)
        self._index += 1


class OrbitVideoVisualizer(Visualizer):
    """Produces a video where the camera orbits the render volume."""

    def __init__(self, results_dir: str, num_steps: int,
                 resolution: Resolution, num_frames: int,
                 num_samples: int, color_space: str):
        """Constructor.

        Args:
            results_dir (str): the base results directory
            num_steps (int): the number of steps in the training sequence
            resolution (Resolution): the resolution of the video
            num_frames (int): number of frames in the video
            num_samples (int): number of samples per ray
            color_space (str): the color space (RGB or YCrCb)
        """
        video_dir = os.path.join(results_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        self._output_dir = video_dir
        cameras = orbit(np.array([0, 1, 0]), np.array([0, 0, -1]),
                        num_frames, 40, resolution.square(), 4)
        bounds = np.eye(4, dtype=np.float32) * 2
        self._sampler = RaySampler(bounds, cameras, num_samples)
        self._interval = num_steps // num_frames
        self._index = 0
        self._color_space = color_space

    def visualize(self, step: int, render: ImageRender,
                  _: ActivationRender):
        """Create a visualization using the provided render functions.

        Args:
            step (int): Step in the optimization
            render (ImageRender): Render function in image space
            act_render (ActivationRender): Render function for the activations
        """
        if step % self._interval != 0:
            return

        camera = self._index % self._sampler.num_cameras
        samples = self._sampler.rays_for_camera(camera)
        pred = render(samples, False)
        image = self._sampler.to_image(camera, pred.color, self._color_space)
        name = "frame_{:05d}.png".format(self._index)
        path = os.path.join(self._output_dir, name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        self._index += 1


class ActivationVisualizer(Visualizer):
    """Creates a video of the layer activations during training."""

    def __init__(self, results_dir: str, num_steps: int,
                 resolution: Resolution, num_frames: int,
                 num_samples: int, color_space: str):
        """Constructor.

        Args:
            results_dir (str): the base results directory
            num_steps (int): the number of steps in the training sequence
            resolution (Resolution): the resolution of the video
            num_frames (int): number of frames in the video
            num_samples (int): number of samples per ray
            color_space (str): the color space (RGB or YCrCb)
        """
        act_dir = os.path.join(results_dir, "activations")
        os.makedirs(act_dir, exist_ok=True)
        self._output_dir = act_dir
        cameras = orbit(np.array([0, 1, 0]), np.array([0, 0, -1]),
                        num_frames, 40, resolution.square(), 4)
        bounds = np.eye(4, dtype=np.float32) * 2
        self._sampler = RaySampler(bounds, cameras, num_samples)
        self._interval = num_steps // num_frames
        self._index = 0
        self._color_space = color_space

    def visualize(self, step: int, _: ImageRender,
                  act_render: ActivationRender):
        """Create a visualization using the provided render functions.

        Args:
            step (int): Step in the optimization
            render (ImageRender): Render function in image space
            act_render (ActivationRender): Render function for the activations
        """
        if step % self._interval != 0:
            return

        image = act_render(self._sampler, self._index)
        name = "frame_{:05d}.png".format(self._index)
        path = os.path.join(self._output_dir, name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
        self._index += 1


class ComparisonVisualizer(Visualizer):
    """This visualizer compares training and validation renders."""

    def __init__(self, results_dir: str, num_steps: int,
                 num_frames: int,
                 train: ImageDataset, val: ImageDataset):
        """Constructor.

        Args:
            results_dir (str): the base results directory
            num_steps (int): the number of steps in the training sequence
            num_frames (int): number of frames in the video
            train (ImageDataset): training data
            val (ImageDataset): validation data
        """
        compare_dir = os.path.join(results_dir, "compare")
        os.makedirs(compare_dir, exist_ok=True)
        assert train.num_cameras == val.num_cameras
        self._output_dir = compare_dir
        self._train = train
        self._val = val
        self._interval = num_steps // num_frames
        self._index = 0

    def visualize(self, step: int, render: ImageRender,
                  _: ActivationRender):
        """Create a visualization using the provided render functions.

        Args:
            step (int): Step in the optimization
            render (ImageRender): Render function in image space
            act_render (ActivationRender): Render function for the activations
        """
        if step % self._interval != 0:
            return

        num_cameras = self._train.num_cameras
        resolution = self._train.cameras[0].resolution
        width = resolution.width * 4
        height = resolution.height * num_cameras
        frame = np.zeros((height, width, 3), np.uint8)
        c = [i * resolution.width for i in range(5)]
        for camera in range(num_cameras):
            r0 = camera * resolution.height
            r1 = r0 + resolution.height
            samples = self._train.rays_for_camera(camera)
            act = self._train.render(samples)
            pred = render(samples, False)
            frame[r0:r1, c[0]:c[1]] = self._train.to_image(camera, act.color)
            frame[r0:r1, c[1]:c[2]] = self._train.to_image(camera, pred.color)

            samples = self._val.rays_for_camera(camera)
            act = self._val.render(samples)
            pred = render(samples, False)
            frame[r0:r1, c[2]:c[3]] = self._val.to_image(camera, act.color)
            frame[r0:r1, c[3]:c[4]] = self._val.to_image(camera, pred.color)

        name = "frame_{:05d}.png".format(self._index)
        path = os.path.join(self._output_dir, name)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame)
        self._index += 1
