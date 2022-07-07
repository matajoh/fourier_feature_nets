"""Module implementing a differentiable volumetric raycaster."""

import copy
import time
from typing import List, NamedTuple, OrderedDict

import cv2
from matplotlib.pyplot import get_cmap
import numpy as np
import scenepic as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from azureml.core import Run
except ImportError:
    Run = None
    print("Unable to import AzureML, running as local experiment")

from .ray_dataset import RayDataset
from .ray_sampler import RaySampler, RaySamples
from .utils import (
    calculate_blend_weights,
    ETABar,
    exponential_lr_decay,
    RenderResult
)
from .visualizers import Visualizer

LogEntry = NamedTuple("LogEntry", [("step", int), ("timestamp", float),
                                   ("state", OrderedDict[str, torch.Tensor]),
                                   ("train_psnr", float), ("val_psnr", float)])


class Raycaster(nn.Module):
    """Implementation of a volumetric raycaster."""

    def __init__(self, model: nn.Module):
        """Constructor.

        Args:
            model (nn.Module): The model used to predict color and opacity.
            use_view (bool, optional): Whether to pass view information to
                                       the model. Defaults to False.
        """
        nn.Module.__init__(self)
        self.model = model

    def render(self, ray_samples: RaySamples,
               include_depth=False) -> RenderResult:
        """Render the ray samples.

        Args:
            ray_samples (RaySamples): The ray samples to render.
            include_depth (bool, optional): Whether to render depth.
                                            Defaults to False.

        Returns:
            RenderResult: The per-ray rendering result.
        """
        num_rays, num_samples = ray_samples.positions.shape[:2]
        positions = ray_samples.positions.reshape(-1, 3)
        if self.model.use_view:
            views = ray_samples.view_directions.reshape(-1, 3)
            color_o = self.model(positions, views)
        else:
            color_o = self.model(positions)

        color_o = color_o.reshape(num_rays, num_samples, 4)
        color, opacity = torch.split(color_o, [3, 1], -1)
        color = torch.sigmoid(color)
        opacity = F.softplus(opacity)

        opacity = opacity.squeeze(-1)
        weights = calculate_blend_weights(ray_samples.t_values, opacity)

        output_color = weights.unsqueeze(-1) * color
        output_color = output_color.sum(-2)

        weights = weights[:, :-1]
        output_alpha = weights.sum(-1)

        if include_depth:
            cutoff = weights.argmax(-1)
            cutoff[output_alpha < .1] = -1
            output_depth = ray_samples.t_values[torch.arange(num_rays),
                                                cutoff]
        else:
            output_depth = None

        return RenderResult(output_color, output_alpha, output_depth)

    def _loss(self, dataset: RayDataset, rays: RaySamples) -> torch.Tensor:
        device = next(self.model.parameters()).device
        rays = rays.to(device)

        render = self.render(rays, True)
        return dataset.loss(rays, render)

    def batched_render(self, samples: RaySamples,
                       batch_size: int, include_depth: bool) -> RenderResult:
        self.model.eval()
        colors = []
        alphas = []
        depths = []
        with torch.no_grad():
            device = next(self.model.parameters()).device
            num_rays = len(samples.positions)
            for start in range(0, num_rays, batch_size):
                end = min(start + batch_size, num_rays)
                idx = list(range(start, end))
                batch = samples.subset(idx)
                batch = batch.to(device)
                pred = self.render(batch, include_depth).numpy()
                colors.append(pred.color)
                alphas.append(pred.alpha)
                if include_depth:
                    depths.append(pred.depth)

        self.model.train()
        return RenderResult(
            np.concatenate(colors),
            np.concatenate(alphas),
            np.concatenate(depths) if include_depth else None
        )

    def render_image(self, sampler: RaySampler,
                     index: int,
                     batch_size: int,
                     color_space="RGB") -> np.ndarray:
        """Renders an image using the ray caster.

        Args:
            sampler (RaySampler): used to produce ray samples
            index (int): Index. Will be used to determine the camera to sample.
            batch_size (int): Number of rays per batch.
            color_space (str, optional): The color space used by the model.
                                         Defaults to "RGB".

        Returns:
            np.ndarray: a (H,W,3) RGB image.
        """
        camera = index % sampler.num_cameras
        samples = sampler.rays_for_camera(camera)
        pred = self.batched_render(samples, batch_size, False)
        return sampler.to_image(camera, pred.color, color_space)

    def render_activations(self, sampler: RaySampler,
                           index: int,
                           batch_size: int,
                           color_space="RGB") -> np.ndarray:
        """Renders an activation grid image."""
        camera = index % sampler.num_cameras
        self.model.eval()
        self.model.keep_activations = True
        with torch.no_grad():
            device = next(self.model.parameters()).device
            samples = sampler.rays_for_camera(camera)
            samples = samples.to(device)
            num_rays = len(samples.positions)
            palette = self.model.layers[-1].weight.data.detach()
            bias = self.model.layers[-1].bias.data.detach()
            activation_values = []
            for start in range(0, num_rays, batch_size):
                end = min(start + batch_size, num_rays)
                idx = list(range(start, end))
                batch = samples.subset(idx)
                batch = batch.to(device)
                self.model(batch.positions)
                activation = np.transpose(self.model.activations[-1], [2, 0, 1])
                activation = torch.from_numpy(activation).unsqueeze(-1).to(device)
                palette = torch.transpose(palette, 0, 1).reshape(-1, 1, 1, 4)
                activation_values.append(activation * palette + bias)

        self.model.train()
        self.model.keep_activations = False
        activation_values = torch.cat(activation_values, dim=1)

        num_grid = 8
        grid_index = np.arange(num_grid*num_grid)
        grid_size = sampler.image_width
        size = grid_size * num_grid
        act_pixels = np.zeros((size, size, 3), np.uint8)
        for i in range(num_grid):
            rstart = i * grid_size
            rend = rstart + grid_size
            for j in range(num_grid):
                cstart = j * grid_size
                cend = cstart + grid_size
                color_o = activation_values[grid_index[i*num_grid + j]]
                color, opacity = torch.split(color_o, [3, 1], -1)
                color = torch.sigmoid(color)
                opacity = F.softplus(opacity)

                opacity = opacity.squeeze(-1)
                weights = calculate_blend_weights(samples.t_values, opacity)

                color = weights.unsqueeze(-1) * color
                color = color.sum(-2)
                color = color.cpu().numpy()
                pixels = sampler.to_image(camera, color, color_space)

                act_pixels[rstart:rend, cstart:cend] = pixels

        return act_pixels

    def _validate(self,
                  dataset: RayDataset,
                  batch_size: int,
                  step: int) -> torch.Tensor:
        loss = []
        num_rays = len(dataset)
        num_validate_rays = min(num_rays, 1024*100)
        if num_validate_rays < num_rays:
            val_index = np.linspace(0, num_rays, num_validate_rays, endpoint=False)
            val_index = val_index.astype(np.int32)
            val_index = dataset.to_valid(val_index.tolist())
        else:
            val_index = np.arange(num_rays)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, num_validate_rays, batch_size):
                if start + batch_size > len(val_index):
                    break

                batch = val_index[start:start + batch_size]
                batch_rays = dataset.get_rays(batch, step)
                loss.append(self._loss(dataset, batch_rays).item())

        self.model.train()
        loss = np.mean(loss)
        psnr = -10. * np.log10(loss)
        return psnr

    def fit(self, train_dataset: RayDataset,
            val_dataset: RayDataset,
            batch_size: int,
            learning_rate: float,
            num_steps: int,
            crop_steps: int,
            report_interval: int,
            decay_rate: float,
            decay_steps: int,
            weight_decay: float,
            visualizers: List[Visualizer],
            disable_aml=False) -> List[LogEntry]:
        """Fits the volumetric model using the raycaster.

        Args:
            train_dataset (RayDataset): The train dataset.
            val_dataset (RayDataset): The validation dataset.
            batch_size (int): The ray batch size.
            learning_rate (float): Initial learning rate for the model.
            num_steps (int): Number of steps (i.e. batches) to use for training.
            crop_steps (int): Number of steps to use center-cropped data at
                              the beginning of training.
            report_interval (int): Frequency for progress reports
            decay_rate (float): Exponential decay term for the learning rate
            decay_steps (int): Number of steps over which the exponential decay
                               is compounded.
            visualizers (List[Visualizer]): List of visualizer objects

        Returns:
            List[LogEntry]: logging information from the training run
        """
        if Run and not disable_aml:
            run = Run.get_context()
        else:
            run = None

        trainval_dataset = train_dataset.sample_cameras(val_dataset.num_cameras,
                                                        val_dataset.num_samples,
                                                        False)

        optim = torch.optim.Adam(self.model.parameters(), learning_rate,
                                 weight_decay=weight_decay)
        step = 0
        start_time = time.time()
        log = []
        epoch = 0
        dataset_mode = train_dataset.mode
        if crop_steps:
            train_dataset.mode = RayDataset.Mode.Center
            val_dataset.mode = RayDataset.Mode.Center
            trainval_dataset.mode = RayDataset.Mode.Center
        else:
            val_dataset.mode = dataset_mode
            trainval_dataset.mode = dataset_mode

        def render_image(samples: RaySamples, include_depth: bool):
            return self.batched_render(samples, batch_size, include_depth)

        def render_act(sampler: RaySampler, camera: int):
            return self.render_activations(sampler, camera, batch_size,
                                           train_dataset.color_space)

        while step <= num_steps:
            num_rays = len(train_dataset)
            index = np.arange(num_rays)
            np.random.shuffle(index)

            for start in range(0, num_rays, batch_size):
                if step > num_steps:
                    break

                exponential_lr_decay(optim, learning_rate, step,
                                     decay_rate, decay_steps)
                end = min(start + batch_size, num_rays)
                batch = index[start:end].tolist()
                batch_rays = train_dataset.get_rays(batch, step)
                optim.zero_grad()
                loss = self._loss(train_dataset, batch_rays)
                loss.backward()
                optim.step()

                if step < 10 or step % report_interval == 0:
                    epoch += 1
                    train_psnr = self._validate(trainval_dataset, batch_size,
                                                step)
                    val_psnr = self._validate(val_dataset, batch_size, step)
                    current_lr = optim.param_groups[0]["lr"]
                    current_time = time.time()
                    if step >= report_interval:
                        time_per_step = (current_time - start_time) / step
                        remaining_time = (num_steps - step) * time_per_step
                        eta = time.gmtime(current_time + remaining_time)
                        eta = time.strftime("%a, %d %b %Y %H:%M:%S +0000", eta)
                    else:
                        time_per_step = 0
                        eta = "N/A"

                    print("{:07}".format(step),
                          "{:2f} s/step".format(time_per_step),
                          "psnr_train: {:2f}".format(train_psnr),
                          "val_psnr: {:2f}".format(val_psnr),
                          "lr: {:.2e}".format(current_lr),
                          "eta:", eta)

                    if run:
                        run.log("psnr_train", train_psnr)
                        run.log("psnr_val", val_psnr)
                        run.log("time_per_step", time_per_step)

                    if step % report_interval == 0:
                        state_dict = copy.deepcopy(self.model.state_dict())
                        log.append(LogEntry(step, current_time - start_time,
                                            state_dict, train_psnr, val_psnr))

                    if train_dataset.mode == RayDataset.Mode.Center and step >= crop_steps:
                        print("Removing center crop...")
                        train_dataset.mode = dataset_mode
                        val_dataset.mode = dataset_mode
                        trainval_dataset.mode = dataset_mode
                        step += 1
                        break

                for visualizer in visualizers:
                    visualizer.visualize(step, render_image, render_act)

                step += 1

        return log

    def to_scenepic(self, dataset: RayDataset, num_cameras=10,
                    resolution=50, num_samples=64,
                    empty_threshold=0.1) -> sp.Scene:
        """Creates a scenepic displaying the state of the volumetric model.

        Args:
            dataset (RayDataset): The dataset to use for visualization.
            num_cameras (int, optional): Number of cameras to show.
                                         Defaults to 10.
            resolution (int, optional): Resolution to use for ray sampling.
                                        Defaults to 50.
            num_samples (int, optional): Number of samples per ray.
                                         Defaults to 64.
            empty_threshold (float, optional): Opacity threshold to determine if
                                               a sample is "empty".
                                               Defaults to 0.1.

        Returns:
            sp.Scene: The constructed scenepic
        """
        dataset = dataset.sample_cameras(num_cameras, num_samples, False)

        scene = sp.Scene()
        frustums = scene.create_mesh("frustums", layer_id="frustums")
        canvas_res = dataset.cameras[0].resolution.scale_to_height(800)
        canvas = scene.create_canvas_3d(width=canvas_res.width,
                                        height=canvas_res.height)
        canvas.shading = sp.Shading(sp.Colors.Gray)

        cmap = get_cmap("jet")
        camera_colors = cmap(np.linspace(0, 1, len(dataset.cameras)))[:, :3]
        image_meshes = []
        bar = ETABar("Adding cameras", max=dataset.num_cameras)
        for pixels, camera, color in zip(dataset.images, dataset.cameras,
                                         camera_colors):
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

        bar.finish()

        image_res = dataset.cameras[0].resolution
        sample_res = image_res.scale_to_height(resolution)
        x_vals = np.linspace(0, image_res.width - 1, sample_res.width) + 0.5
        y_vals = np.linspace(0, image_res.height - 1, sample_res.height) + 0.5
        x_vals, y_vals = np.meshgrid(x_vals.astype(np.int32),
                                     y_vals.astype(np.int32))
        index = y_vals.reshape(-1) * image_res.width + x_vals.reshape(-1)
        dataset.subsample_index = set(index.tolist())

        bar = ETABar("Sampling rays", max=dataset.num_cameras)
        device = next(self.model.parameters()).device
        for i, camera in enumerate(dataset.cameras):
            bar.next()
            ray_samples = dataset.rays_for_camera(i)
            ray_samples = ray_samples.to(device)

            with torch.no_grad():
                positions = ray_samples.positions.reshape(-1, 3)
                if self.model.use_view:
                    views = ray_samples.view_directions.reshape(-1, 3)
                    color_o = self.model(positions, views)
                else:
                    color_o = self.model(positions)

                color_o = color_o.reshape(-1, num_samples, 4)
                color, opacity = torch.split(color_o, [3, 1], -1)
                color = torch.sigmoid(color)
                opacity = F.softplus(opacity)

            positions = ray_samples.positions.cpu().numpy().reshape(-1, 3)
            color = color.cpu().numpy().reshape(-1, 3)
            opacity = opacity.reshape(-1).cpu().numpy()

            empty = opacity < empty_threshold
            not_empty = np.logical_not(empty)

            samples = scene.create_mesh()
            samples.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.02))
            samples.enable_instancing(positions=positions[not_empty],
                                      colors=color[not_empty])

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

        dataset.subsample_index = None

        scene.framerate = 10
        return scene
