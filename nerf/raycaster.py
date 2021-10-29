import os
import sys
import time

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


class Raycaster(nn.Module):
    def __init__(self, model: nn.Module, use_view=False):
        nn.Module.__init__(self)
        self.model = model
        self._use_alpha = False
        self._use_view = use_view

    def render(self, ray_samples: RaySamples,
               include_depth=False) -> torch.Tensor:
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

        deltas = ray_samples.t_values[:, 1:] - ray_samples.t_values[:, :-1]
        max_dist = torch.full_like(deltas[:, :1], 1e10)
        deltas = torch.cat([deltas, max_dist], dim=-1)
        deltas = deltas.unsqueeze(-1)

        alpha = 1 - torch.exp(-(opacity * deltas))
        ones = torch.ones_like(alpha)

        trans = torch.minimum(ones, 1 - alpha + 1e-10)
        trans, _ = trans.split([num_samples - 1, 1], dim=-2)
        trans = torch.cat([torch.ones_like(trans[:, :1, :]), trans], dim=-2)
        trans = torch.cumprod(trans, -2)
        weights = alpha * trans
        output_color = weights * color
        output_color = output_color.sum(-2)

        weights = weights[:, :-1, 0]
        output_alpha = weights.sum(-1)

        if include_depth:
            cutoff = weights.argmax(-1)
            cutoff[output_alpha < .1] = -1
            output_depth = ray_samples.t_values[torch.arange(num_rays),
                                                cutoff]
        else:
            output_depth = None

        return output_color, output_alpha, output_depth

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
            resolution = dataset.resolution
            num_rays = resolution * resolution
            image_start = index * num_rays
            image_end = image_start + num_rays
            predicted = []
            actual = []
            depth = []
            error = []
            loss = []
            max_depth = 10
            for start in range(image_start, image_end, batch_size):
                end = min(start + batch_size, image_end)
                idx = list(range(start, end))
                ray_samples = dataset[idx]
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

            predicted = np.concatenate(predicted)
            predicted = np.clip(predicted, 0, 1)
            predicted = predicted.reshape(resolution, resolution, 3)
            predicted = (predicted * 255).astype(np.uint8)

            actual = np.concatenate(actual)
            actual = actual.reshape(resolution, resolution, 3)
            actual = (actual * 255).astype(np.uint8)

            depth = np.concatenate(depth)
            depth = np.clip(depth, 0, max_depth)
            depth = (depth / max_depth).reshape(resolution, resolution, 1)
            depth = (depth * 255).astype(np.uint8)

            error = np.concatenate(error)
            error = np.sqrt(error)
            error = (error / error.max()).reshape(resolution, resolution, 1)
            error = (error * 255).astype(np.uint8)

            name = "s{:07}_c{:03}.png".format(step, index)
            image_dir = os.path.join(results_dir, dataset.label)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            image_path = os.path.join(image_dir, name)

            compare = np.zeros((resolution*2, resolution*2, 3), np.uint8)
            compare[:resolution, :resolution] = predicted
            compare[resolution:, :resolution] = actual
            compare[:resolution, resolution:] = depth
            compare[resolution:, resolution:] = error
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
        if Run:
            run = Run.get_context()
        else:
            run = None

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self._use_alpha = train_dataset.alphas is not None
        trainval_dataset = train_dataset.sample_cameras(val_dataset.num_cameras)

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

                if step % reporting_interval == 0:
                    val_psnr = self._render_eval_image(val_dataset, step,
                                                       4 * batch_size,
                                                       results_dir, render_index)
                    train_psnr = self._render_eval_image(trainval_dataset, step,
                                                         4 * batch_size,
                                                         results_dir, render_index)
                    current_time = time.time()
                    time_per_step = (current_time - timestamp) / reporting_interval
                    timestamp = current_time
                    print("{:07}".format(step),
                          "{:2f} s/step".format(time_per_step),
                          "psnr_train: {:2f}".format(train_psnr),
                          "loss_train: {:2f}".format(loss.item()),
                          "val_psnr: {:2f}".format(val_psnr))

                    if run:
                        run.log("psnr_train", train_psnr)
                        run.log("loss_train", loss.item())
                        run.log("psnr_val", val_psnr)
                        run.log("time_per_step", time_per_step)

                    log.append((step, timestamp - start_time, train_psnr, val_psnr))
                    render_index += 1

                step += 1

            epoch += 1

        return log

    def to_scenepic(self, dataset, num_cameras=10,
                    resolution=50, num_samples=64) -> sp.Scene:
        dataset = RaySamplingDataset("scenepic",
                                     dataset.images[:num_cameras],
                                     dataset.cameras[:num_cameras],
                                     num_samples,
                                     resolution)

        scene = sp.Scene()
        frustums = scene.create_mesh("frustums", layer_id="frustums")
        canvas = scene.create_canvas_3d(width=800,
                                        height=800)
        canvas.shading = sp.Shading(sp.Colors.Gray)

        cmap = get_cmap("jet")
        camera_colors = cmap(np.linspace(0, 1, len(dataset.cameras)))[:, :3]
        image_meshes = []
        sys.stdout.write("Adding cameras")
        for pixels, camera, color in zip(dataset.images, dataset.cameras,
                                         camera_colors):
            sys.stdout.write(".")
            sys.stdout.flush()
            camera = camera.to_scenepic()

            image = scene.create_image()
            pixels = cv2.resize(pixels, (200, 200), cv2.INTER_AREA)
            image.from_numpy(pixels[..., :3])
            mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id,
                                     double_sided=True)
            mesh.add_camera_image(camera, depth=0.5)
            image_meshes.append(mesh)

            frustums.add_camera_frustum(camera, color, depth=0.5, thickness=0.01)

        print("done.")

        sys.stdout.write("Sampling rays...")
        num_rays = dataset.resolution ** 2
        device = next(self.model.parameters()).device
        for i, camera in enumerate(dataset.cameras):
            sys.stdout.write(".")
            sys.stdout.flush()
            start = i * num_rays
            end = start + num_rays
            index = [i for i in range(start, end)]
            ray_samples = dataset[index]
            ray_samples = ray_samples.to(device)

            with torch.no_grad():
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

            positions = ray_samples.positions.cpu().numpy().reshape(-1, 3)
            color = color.cpu().numpy().reshape(-1, 3)
            opacity = opacity.reshape(-1).cpu().numpy()

            empty = opacity < 0.1
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

        print("done.")

        scene.framerate = 10
        return scene
