"""Module implementing a differentiable volumetric raycaster."""

import os
import time
from typing import NamedTuple

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

from .datasets import RaySamples, RaySamplingDataset
from .utils import calculate_blend_weights, ETABar


RenderResult = NamedTuple("RenderResult", [("color", torch.Tensor),
                                           ("alpha", torch.Tensor),
                                           ("depth", torch.Tensor)])
LogEntry = NamedTuple("LogEntry", [("step", int), ("timestamp", float),
                                   ("train_psnr", float), ("val_psnr", float)])


class Raycaster(nn.Module):
    """Implementation of a volumetric raycaster."""

    def __init__(self, model: nn.Module, use_view=False):
        """Constructor.

        Args:
            model (nn.Module): The model used to predict color and opacity.
            use_view (bool, optional): Whether to pass view information to
                                       the model. Defaults to False.
        """
        nn.Module.__init__(self)
        self.model = model
        self._use_alpha = False
        self._use_view = use_view

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
        if self._use_view:
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

    def _loss(self, ray_samples: RaySamples) -> torch.Tensor:
        device = next(self.model.parameters()).device
        ray_samples = ray_samples.to(device)

        colors, alphas, _ = self.render(ray_samples)
        loss = (colors - ray_samples.colors).square().mean()
        if self._use_alpha:
            loss += (alphas - ray_samples.alphas).square().mean()

        return loss

    def _render_eval_image(self, dataset: RaySamplingDataset, step: int,
                           batch_size: int, results_dir: str,
                           index: int):
        with torch.no_grad():
            index = index % dataset.num_cameras
            device = next(self.model.parameters()).device
            image_rays = dataset.rays_for_camera(index)
            num_rays = len(image_rays.positions)
            predicted = []
            actual = []
            depth = []
            error = []
            loss = []
            max_depth = 10
            for start in range(0, num_rays, batch_size):
                end = min(start + batch_size, num_rays)
                idx = list(range(start, end))
                ray_samples = image_rays.subset(idx)
                ray_samples = ray_samples.to(device)
                pred_colors, pred_alphas, pred_depth = self.render(ray_samples, True)
                pred_colors = pred_colors.cpu().numpy()
                act_colors = ray_samples.colors.cpu().numpy()
                pred_error = np.square(act_colors - pred_colors).sum(-1) / 3
                if self._use_alpha:
                    pred_alphas = pred_alphas.cpu().numpy()
                    act_alphas = ray_samples.alphas.cpu().numpy()
                    pred_error = 3 * pred_error
                    pred_error += np.square(act_alphas - pred_alphas)
                    pred_error /= 4

                loss.append(np.mean(pred_error).item())
                predicted.append(pred_colors)
                actual.append(act_colors)
                depth.append(pred_depth.clamp(0, max_depth).cpu().numpy())
                error.append(pred_error)

            loss = np.mean(loss)
            psnr = -10. * np.log10(loss)

            width, height = dataset.image_width, dataset.image_height
            predicted = np.concatenate(predicted)
            predicted = np.clip(predicted, 0, 1)
            predicted = predicted.reshape(height, width, 3)
            predicted = (predicted * 255).astype(np.uint8)

            actual = np.concatenate(actual)
            actual = actual.reshape(height, width, 3)
            actual = (actual * 255).astype(np.uint8)

            depth = np.concatenate(depth)
            depth = np.clip(depth, 0, max_depth)
            depth = (depth / max_depth).reshape(height, width, 1)
            depth = (depth * 255).astype(np.uint8)

            error = np.concatenate(error)
            error = np.sqrt(error)
            error = (error / error.max()).reshape(height, width, 1)
            error = (error * 255).astype(np.uint8)

            name = "s{:07}_c{:03}.png".format(step, index)
            image_dir = os.path.join(results_dir, dataset.label)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            image_path = os.path.join(image_dir, name)

            compare = np.zeros((height*2, width*2, 3), np.uint8)
            compare[:height, :width] = predicted
            compare[height:, :width] = actual
            compare[:height, width:] = depth
            compare[height:, width:] = error
            cv2.imwrite(image_path, compare[:, :, ::-1])
            return psnr

    def fit(self, train_dataset: RaySamplingDataset,
            val_dataset: RaySamplingDataset,
            results_dir: str,
            batch_size: int,
            learning_rate: float,
            num_steps: int,
            reporting_interval: int,
            crop_epochs: int):
        """Fits the volumetric model using the raycaster.

        Args:
            train_dataset (RaySamplingDataset): The train dataset.
            val_dataset (RaySamplingDataset): The validation dataset.
            results_dir (str): The output directory for results images.
            batch_size (int): The ray batch size.
            learning_rate (float): Learning rate for the model.
            num_steps (int): Number of steps (i.e. batches) to use for training.
            reporting_interval (int): Number of steps between logging and images
            crop_epochs (int): Number of epochs to use center-cropped data at
                               the beginning of training.

        Returns:
            List[LogEntry]: logging information from the training run
        """
        if Run:
            run = Run.get_context()
        else:
            run = None

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self._use_alpha = train_dataset.alphas is not None
        trainval_dataset = train_dataset.sample_cameras(val_dataset.num_cameras,
                                                        val_dataset.num_samples,
                                                        False)

        optim = torch.optim.Adam(self.model.parameters(), learning_rate)
        step = 0
        start_time = time.time()
        timestamp = start_time
        log = []
        epoch = 0
        render_index = 0
        while step < num_steps:
            train_dataset.center_crop = epoch < crop_epochs
            num_rays = len(train_dataset)
            print("Epoch", epoch,
                  " -- center_crop:", epoch < crop_epochs,
                  "num_rays:", num_rays)
            index = np.arange(num_rays)
            np.random.shuffle(index)

            for start in range(0, num_rays, batch_size):
                if step == num_steps:
                    break

                end = min(start + batch_size, num_rays)
                batch = index[start:end].tolist()
                ray_samples = train_dataset[batch]
                optim.zero_grad()
                loss = self._loss(ray_samples)
                loss.backward()
                optim.step()

                if step and step % reporting_interval == 0:
                    val_psnr = self._render_eval_image(val_dataset, step,
                                                       4 * batch_size,
                                                       results_dir, render_index)
                    train_psnr = self._render_eval_image(trainval_dataset, step,
                                                         4 * batch_size,
                                                         results_dir, render_index)
                    current_time = time.time()
                    time_per_step = (current_time - timestamp) / reporting_interval
                    remaining_time = (num_steps - step) * time_per_step
                    eta = time.gmtime(current_time + remaining_time)
                    eta = time.strftime("%a, %d %b %Y %H:%M:%S +0000", eta)
                    timestamp = current_time
                    print("{:07}".format(step),
                          "{:2f} s/step".format(time_per_step),
                          "psnr_train: {:2f}".format(train_psnr),
                          "loss_train: {:2f}".format(loss.item()),
                          "val_psnr: {:2f}".format(val_psnr),
                          "eta:", eta)

                    if run:
                        run.log("psnr_train", train_psnr)
                        run.log("loss_train", loss.item())
                        run.log("psnr_val", val_psnr)
                        run.log("time_per_step", time_per_step)

                    log.append(LogEntry(step, timestamp - start_time,
                                        train_psnr, val_psnr))
                    render_index += 1

                step += 1

            epoch += 1

        return log

    def to_scenepic(self, dataset: RaySamplingDataset, num_cameras=10,
                    resolution=50, num_samples=64,
                    empty_threshold=0.1) -> sp.Scene:
        """Creates a scenepic displaying the state of the volumetric model.

        Args:
            dataset (RaySamplingDataset): The dataset to use for visualization.
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
        height = 800
        width = dataset.image_width * height / dataset.image_height
        canvas = scene.create_canvas_3d(width=width, height=height)
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

        num_x_samples = resolution * dataset.image_width // dataset.image_height
        num_y_samples = resolution
        x_vals = np.linspace(0, dataset.image_width - 1, num_x_samples) + 0.5
        y_vals = np.linspace(0, dataset.image_height - 1, num_y_samples) + 0.5
        x_vals, y_vals = np.meshgrid(x_vals.astype(np.int32),
                                     y_vals.astype(np.int32))
        index = y_vals.reshape(-1) * dataset.image_width + x_vals.reshape(-1)
        index = index.tolist()

        bar = ETABar("Sampling rays", max=dataset.num_cameras)
        device = next(self.model.parameters()).device
        for i, camera in enumerate(dataset.cameras):
            bar.next()
            ray_samples = dataset.rays_for_camera(i)
            ray_samples = ray_samples.subset(index)
            ray_samples = ray_samples.to(device)

            with torch.no_grad():
                positions = ray_samples.positions.reshape(-1, 3)
                if self._use_view:
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

        scene.framerate = 10
        return scene
