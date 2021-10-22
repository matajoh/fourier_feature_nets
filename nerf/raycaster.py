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

from .datasets import RaySamples, RaySamplingDataset


class Raycaster(nn.Module):
    def __init__(self, train_dataset: RaySamplingDataset,
                 val_dataset: RaySamplingDataset,
                 model: nn.Module, results_dir: str):
        nn.Module.__init__(self)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._results_dir = results_dir
        self._val_index = 0

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def render(self, ray_samples: RaySamples) -> torch.Tensor:
        num_rays, num_samples = ray_samples.deltas.shape[:2]
        positions = ray_samples.positions.reshape(-1, 3)
        color_o = self.model(positions)
        color_o = color_o.reshape(num_rays, num_samples, 4)
        color, opacity = torch.split(color_o, [3, 1], -1)
        color = torch.sigmoid(color)
        opacity = F.softplus(opacity)

        left_trans = torch.ones((num_rays, 1, 1), dtype=torch.float32)
        left_trans = left_trans.to(positions.device)
        alpha = 1 - torch.exp(-(opacity * ray_samples.deltas))
        ones = torch.ones_like(alpha)
        trans = torch.minimum(ones, 1 - alpha + 1e-10)
        _, trans = trans.split([1, num_samples - 1], dim=-2)
        trans = torch.cat([left_trans, trans], -2)
        weights = alpha * torch.cumprod(trans, -2)
        output_color = weights * color
        output_color = output_color.sum(-2)

        output_depth = weights.squeeze(-1) * ray_samples.t_values
        output_depth = output_depth.sum(-1)
        return output_color, output_depth

    def _loss(self, ray_samples: RaySamples) -> torch.Tensor:
        device = next(self.model.parameters()).device
        ray_samples = ray_samples.to(device)
        colors, _ = self.render(ray_samples)
        loss = (colors - ray_samples.colors).square().mean()
        return loss

    def _val_image(self, step: int, batch_size: int):
        with torch.no_grad():
            device = next(self.model.parameters()).device
            resolution = self.val_dataset.resolution
            num_rays = resolution * resolution
            image_start = self._val_index * num_rays
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
                ray_samples = self.val_dataset[idx]
                ray_samples = ray_samples.to(device)
                pred_colors, pred_depth = self.render(ray_samples)
                pred_colors = pred_colors.detach().cpu().numpy()
                act_colors = ray_samples.colors.cpu().numpy()
                pred_error = np.square(act_colors - pred_colors)
                loss.append(np.mean(pred_error).item())
                predicted.append(pred_colors)
                actual.append(act_colors)
                depth.append(pred_depth.clamp(0, max_depth).cpu().numpy())
                error.append(pred_error)

            loss = np.mean(loss)
            psnr = -10. * np.log10(loss)

            predicted = np.concatenate(predicted)
            predicted = predicted.reshape(resolution, resolution, 3)
            predicted = (predicted * 255).astype(np.uint8)

            actual = np.concatenate(actual)
            actual = actual.reshape(resolution, resolution, 3)
            actual = (actual * 255).astype(np.uint8)

            depth = np.concatenate(depth)
            depth = (depth / max_depth).reshape(resolution, resolution, 1)
            depth = (depth * 255).astype(np.uint8)

            error = np.concatenate(error)
            error = np.sqrt(error.sum(-1))
            error = (error / error.max()).reshape(resolution, resolution, 1)
            error = (error * 255).astype(np.uint8)

            name = "val_s{:03}_c{:03}.png".format(step, self._val_index)
            image_path = os.path.join(self._results_dir, name)

            compare = np.zeros((resolution*2, resolution*2, 3), np.uint8)
            compare[:resolution, :resolution] = predicted
            compare[resolution:, :resolution] = actual
            compare[:resolution, resolution:] = depth
            compare[resolution:, resolution:] = error
            cv2.imwrite(image_path, compare[:, :, ::-1])
            self._val_index += 1
            if self._val_index * num_rays == len(self.val_dataset):
                self._val_index = 0

            return psnr

    def fit(self, batch_size: int, learning_rate: float, num_steps: int,
            reporting_interval: int, crop_epochs: int):
        optim = torch.optim.Adam(self.model.parameters(), learning_rate)
        step = 0
        start_time = time.time()
        timestamp = start_time
        log = []
        epoch = 0
        while step < num_steps:
            self.train_dataset.center_crop = epoch < crop_epochs
            num_rays = len(self.train_dataset)
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
                ray_samples = self.train_dataset[batch]
                optim.zero_grad()
                loss = self._loss(ray_samples)
                loss.backward()
                optim.step()

                if step % reporting_interval == 0:
                    psnr = self._val_image(step, batch_size)
                    current_time = time.time()
                    time_per_step = (current_time - timestamp) / reporting_interval
                    timestamp = current_time
                    print("{:07}".format(step),
                          "{:2f} s/step".format(time_per_step),
                          "loss: {:2f}".format(loss.item()),
                          "psnr: {:2f}".format(psnr))

                    log.append((step, timestamp - start_time, psnr))

                step += 1

            epoch += 1

        return log

    def to_scenepic(self, num_cameras=10,
                    resolution=64, num_samples=32) -> sp.Scene:
        dataset = RaySamplingDataset(self.val_dataset.images[:num_cameras],
                                     self.val_dataset.cameras[:num_cameras],
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
            image.from_numpy(pixels)
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
                color_o = self.model(ray_samples.positions)
                color_o = color_o.reshape(num_rays, num_samples, 4)
                color, opacity = torch.split(color_o, [3, 1], -1)
                color = torch.sigmoid(color)
                opacity = F.softplus(opacity)

            positions = ray_samples.positions.cpu().numpy().reshape(-1, 3)
            color = color.cpu().numpy().reshape(-1, 3)
            opacity = opacity.reshape(-1).cpu().numpy()

            empty = opacity < 1e-5
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
