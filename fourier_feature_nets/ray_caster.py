"""Module implementing a differentiable volumetric raycaster."""

import copy
import os
import time
from typing import NamedTuple, OrderedDict

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
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            samples = sampler.rays_for_camera(camera)
            num_rays = len(samples.positions)
            predicted = []
            for start in range(0, num_rays, batch_size):
                end = min(start + batch_size, num_rays)
                idx = list(range(start, end))
                batch = samples.subset(idx)
                batch = batch.to(device)
                pred = self.render(batch, False)
                pred_colors = pred.color.cpu().numpy()
                predicted.append(pred_colors)

        self.model.train()
        predicted = np.concatenate(predicted)
        return sampler.to_image(camera, predicted, color_space)

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

    def _render_act(self, sampler: RaySampler,
                    index: int,
                    color_space: str,
                    batch_size: int,
                    results_dir: str):
        image = self.render_activations(sampler, index, batch_size, color_space)
        act_dir = os.path.join(results_dir, "activations")
        if not os.path.exists(act_dir):
            os.makedirs(act_dir)

        path = os.path.join(act_dir, "frame_{:05d}.png".format(index))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

    def _render_video(self, sampler: RaySampler,
                      index: int,
                      color_space: str,
                      batch_size: int,
                      results_dir: str):
        image = self.render_image(sampler, index, batch_size, color_space)
        video_dir = os.path.join(results_dir, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        path = os.path.join(video_dir, "frame_{:05d}.png".format(index))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)

    def _render_eval_image(self, dataset: RayDataset, step: int,
                           batch_size: int, results_dir: str,
                           index: int):
        camera = index % dataset.num_cameras
        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            image_rays = dataset.rays_for_camera(camera)
            num_rays = len(image_rays.positions)
            predicted = []
            actual = []
            depth = []
            error = []
            max_depth = 10
            for start in range(0, num_rays, batch_size):
                end = min(start + batch_size, num_rays)
                idx = list(range(start, end))
                batch_rays = image_rays.subset(idx)
                batch_rays = batch_rays.to(device)
                pred = self.render(batch_rays, True)
                act = dataset.render(batch_rays)
                pred_colors = pred.color.cpu().numpy()
                act_colors = act.color.cpu().numpy()
                pred_error = np.square(act_colors - pred_colors).sum(-1) / 3
                if act.alpha is not None:
                    pred_alphas = pred.alpha.cpu().numpy()
                    act_alphas = act.alpha.cpu().numpy()
                    pred_error = 3 * pred_error
                    pred_error += np.square(act_alphas - pred_alphas)
                    pred_error /= 4

                predicted.append(pred_colors)
                actual.append(act_colors)
                depth.append(pred.depth.clamp(0, max_depth).cpu().numpy())
                error.append(pred_error)

        self.model.train()

        cam_index = dataset.index_for_camera(camera)

        width, height = dataset.cameras[camera].resolution
        predicted = np.concatenate(predicted)
        predicted_image = dataset.to_image(camera, np.clip(predicted, 0, 1))

        actual_image = np.zeros((height*width, 3), np.float32)
        actual_image[cam_index] = np.concatenate(actual)
        actual_image = actual_image.reshape(height, width, 3)
        actual_image = (actual_image * 255).astype(np.uint8)

        depth_image = np.zeros(height*width, np.float32)
        depth_image[cam_index] = np.concatenate(depth)
        depth_image = np.clip(depth_image, 0, max_depth)
        depth_image = (depth_image / max_depth).reshape(height, width, 1)
        depth_image = (depth_image * 255).astype(np.uint8)

        error_image = np.zeros(height*width, np.float32)
        error_image[cam_index] = np.concatenate(error)
        error_image = np.sqrt(error_image)
        error_image = error_image / error_image.max()
        error_image = error_image.reshape(height, width, 1)
        error_image = (error_image * 255).astype(np.uint8)

        name = "s{:07}_c{:03}.png".format(step, camera)
        image_dir = os.path.join(results_dir, dataset.label)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        image_path = os.path.join(image_dir, name)

        compare = np.zeros((height*2, width*2, 3), np.uint8)
        compare[:height, :width] = predicted_image
        compare[height:, :width] = actual_image
        compare[:height, width:] = depth_image
        compare[height:, width:] = error_image
        compare = cv2.cvtColor(compare, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, compare)

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
            results_dir: str,
            batch_size: int,
            learning_rate: float,
            num_steps: int,
            image_interval: int,
            crop_steps: int,
            report_interval: int,
            decay_rate: float,
            decay_steps: int,
            weight_decay: float,
            video_sampler: RaySampler = None,
            act_sampler: RaySampler = None,
            disable_aml=False):
        """Fits the volumetric model using the raycaster.

        Args:
            train_dataset (RayDataset): The train dataset.
            val_dataset (RayDataset): The validation dataset.
            results_dir (str): The output directory for results images.
            batch_size (int): The ray batch size.
            learning_rate (float): Initial learning rate for the model.
            num_steps (int): Number of steps (i.e. batches) to use for training.
            image_interval (int): Number of steps between logging and images
            crop_steps (int): Number of steps to use center-cropped data at
                              the beginning of training.
            report_interval (int): Frequency for progress reports
            decay_rate (float): Exponential decay term for the learning rate
            decay_steps (int): Number of steps over which the exponential decay
                               is compounded.
            video_sampler (RaySampler, optional): sampler used to create frames
                                                  for a training video.
                                                  Defaults to None.
            act_sampler (RaySampler, optional): sampler used to create
                                                activation images.
                                                Defaults to None.

        Returns:
            List[LogEntry]: logging information from the training run
        """
        if Run and not disable_aml:
            run = Run.get_context()
        else:
            run = None

        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)

        trainval_dataset = train_dataset.sample_cameras(val_dataset.num_cameras,
                                                        val_dataset.num_samples,
                                                        False)

        optim = torch.optim.Adam(self.model.parameters(), learning_rate,
                                 weight_decay=weight_decay)
        step = 0
        start_time = time.time()
        log = []
        epoch = 0
        render_index = 0
        dataset_mode = train_dataset.mode
        if crop_steps:
            train_dataset.mode = RayDataset.Mode.Center
            val_dataset.mode = RayDataset.Mode.Center
            trainval_dataset.mode = RayDataset.Mode.Center
        else:
            val_dataset.mode = dataset_mode
            trainval_dataset.mode = dataset_mode

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

                if results_dir and step % image_interval == 0:
                    if video_sampler or act_sampler:
                        if video_sampler:
                            self._render_video(video_sampler,
                                               render_index,
                                               train_dataset.color_space,
                                               batch_size,
                                               results_dir)

                        if act_sampler:
                            self._render_act(act_sampler,
                                             render_index,
                                             train_dataset.color_space,
                                             batch_size,
                                             results_dir)
                    else:
                        self._render_eval_image(val_dataset, step,
                                                batch_size,
                                                results_dir, render_index)
                        self._render_eval_image(trainval_dataset, step,
                                                batch_size,
                                                results_dir, render_index)

                    render_index += 1

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
