import argparse
import json
import os
import time

import cv2
import nerf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class TinyNeRF(nn.Module):
    def __init__(self, train_dataset: nerf.RaySamplingDataset,
                 val_dataset: nerf.RaySamplingDataset,
                 model: nn.Module, results_dir: str):
        nn.Module.__init__(self)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._results_dir = results_dir
        self._val_index = 0

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def render(self, ray_samples: nerf.RaySamples) -> torch.Tensor:
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

    def _loss(self, ray_samples: nerf.RaySamples) -> torch.Tensor:
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
            cv2.imwrite(image_path, compare)
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


def _load_images(data_dir: str, split: str, num_cameras: int):
    image_dir = os.path.join(data_dir, split)
    images = []
    camera_path = os.path.join(data_dir, "{}_cameras.json".format(split))
    cameras = nerf.CameraInfo.from_json(camera_path)
    cameras = cameras[:num_cameras]
    for camera in cameras:
        image = cv2.imread(os.path.join(image_dir, camera.name + ".png"))
        images.append(image)

    images = np.stack(images)
    return cameras, images


def _parse_args():
    parser = argparse.ArgumentParser("Tiny NeRF")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("nerf_model", choices=["mlp", "basic",
                                               "positional", "gaussian"])
    parser.add_argument("results_dir", help="Path to output results")
    parser.add_argument("--voxels-dir",
                        help="Path to the voxels directory")
    parser.add_argument("--path-length", type=int, default=128,
                        help="Number of voxels to intersect")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=400,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=100,
                        help="Number of cameras")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-channels", type=int, default=256,
                        help="Number of channels in the MLP")
    parser.add_argument("--num-frequencies", type=int, default=256,
                        help="Number of frequencies used for encoding")
    parser.add_argument("--pos-sigma", type=float, default=1.27,
                        help="Value of sigma for the positional model")
    parser.add_argument("--gauss-sigma", type=float, default=6.05,
                        help="Value of sigma for the gaussian model")
    parser.add_argument("--num-steps", type=int, default=50000,
                        help="Number of steps to use for training.")
    parser.add_argument("--report-interval", type=int, default=1000,
                        help="Reporting interval for validation/logging")
    parser.add_argument("--crop-epochs", type=int, default=1,
                        help="Number of epochs to train on center crops")
    return parser.parse_args()


def _main():
    args = _parse_args()

    if args.voxels_dir:
        data = np.load(os.path.join(args.voxels_dir, "carving.npz"))
        voxels = nerf.OcTree.load(data)
        opacity = data["opacity"]
    else:
        voxels = None
        opacity = None

    torch.manual_seed(20080524)
    if args.nerf_model == "mlp":
        model = nerf.MLP(3, 4, args.num_channels)
    elif args.nerf_model == "basic":
        model = nerf.BasicFourierMLP(3, 4, args.num_channels)
    elif args.nerf_model == "positional":
        model = nerf.PositionalFourierMLP(3, 4, args.pos_sigma,
                                          args.num_channels,
                                          args.num_frequencies)
    elif args.nerf_model == "gaussian":
        model = nerf.GaussianFourierMLP(3, 4, args.gauss_sigma,
                                        args.num_channels,
                                        args.num_frequencies)

    train_cameras, train_images = _load_images(args.data_dir, "train",
                                               args.num_cameras)
    train_dataset = nerf.RaySamplingDataset(train_images, train_cameras,
                                            args.num_samples, args.resolution,
                                            args.path_length, voxels, opacity,
                                            stratified=True)

    val_cameras, val_images = _load_images(args.data_dir, "val",
                                           args.num_cameras)
    val_dataset = nerf.RaySamplingDataset(val_images, val_cameras,
                                          args.num_samples, args.resolution,
                                          args.path_length, voxels, opacity,
                                          stratified=False)

    trainer = TinyNeRF(train_dataset, val_dataset, model, args.results_dir)
    trainer.to("cuda")

    log = trainer.fit(args.batch_size, args.learning_rate,
                      args.num_steps, args.report_interval,
                      args.crop_epochs)

    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        json.dump(vars(args), file)
        file.write("\n\n")
        for step, timestamp, psnr in log:
            file.write("{}\t{}\t{}\n".format(step, timestamp, psnr))

    model.save(os.path.join(args.results_dir, "tiny_nerf.model"))


if __name__ == "__main__":
    _main()
