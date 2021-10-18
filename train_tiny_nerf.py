import argparse
import os

import cv2
from nerf import CameraInfo, GaussianFourierMLP, OcTree, RaySamplingDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim


class TinyNeRF(nn.Module):
    def __init__(self, train_dataset: RaySamplingDataset, val_dataset: RaySamplingDataset,
                 model: nn.Module, results_dir: str):
        nn.Module.__init__(self)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self._results_dir = results_dir
        self._val_index = 0

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def render(self,
               positions: torch.Tensor,
               deltas: torch.Tensor) -> torch.Tensor:
        num_rays, num_samples = deltas.shape[:2]
        positions = positions.reshape(-1, 3)
        rgb_o = self.model(positions)
        rgb_o = rgb_o.reshape(num_rays, num_samples, 4)
        rgb, opacity = torch.split(rgb_o, [3, 1], -1)

        left_trans = torch.ones((num_rays, 1, 1), dtype=torch.float32)
        left_trans = left_trans.to(positions.device)
        alpha = 1 - torch.exp(-(opacity * deltas))
        ones = torch.ones_like(alpha)
        trans = torch.minimum(ones, 1 - alpha + 1e-10)
        _, trans = trans.split([1, num_samples - 1], dim=-2)
        trans = torch.cat([left_trans, trans], -2)
        weights = alpha * torch.cumprod(trans, -2)
        outputs = weights * rgb
        outputs = outputs.sum(-2)
        return outputs

    def _loss(self, positions: torch.Tensor,
              deltas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device
        positions = positions.to(device)
        deltas = deltas.to(device)
        targets = targets.to(device)
        outputs = self.render(positions, deltas)
        loss = (outputs - targets).square().sum()
        return loss

    def _val_image(self, epoch: int, step: int, batch_size: int):
        with torch.no_grad():
            device = next(self.model.parameters()).device
            resolution = self.val_dataset.resolution
            num_rays = resolution * resolution
            image_start = self._val_index * num_rays
            image_end = image_start + num_rays
            pixels = []
            for start in range(image_start, image_end, batch_size):
                end = min(start + batch_size, image_end)
                idx = list(range(start, end))
                positions, _, deltas, _ = self.val_dataset[idx]
                output = self.render(positions.to(device), deltas.to(device))
                pixels.append(output.detach().cpu().numpy())

            pixels = np.concatenate(pixels)
            pixels = pixels.reshape(resolution, resolution, 3)
            pixels = (pixels * 255).astype(np.uint8)
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YCrCb2BGR)
            name = "val_e{:04}_s{:03}_c{:03}.png".format(epoch, step,
                                                         self._val_index)
            image_path = os.path.join(self._results_dir, name)
            cv2.imwrite(image_path, pixels)
            self._val_index += 1
            if self._val_index * num_rays == len(self.val_dataset):
                self._val_index = 0
            
    def fit(self, batch_size: int, learning_rate: float, num_epochs: int):
        optim = torch.optim.Adam(self.model.parameters(), learning_rate)
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            num_rays = len(self.train_dataset)
            index = np.arange(num_rays)
            np.random.shuffle(index)

            for step, start in enumerate(range(0, num_rays, batch_size)):
                end = min(start + batch_size, num_rays)
                batch = index[start:end].tolist()
                positions, _, deltas, targets = self.train_dataset[batch]
                optim.zero_grad()
                loss = self._loss(positions, deltas, targets)
                loss.backward()
                optim.step()

                if step % 50 == 0:
                    print(step, "loss:", loss.item())
                    self._val_image(epoch, step, batch_size)


def _load_images(data_dir: str, split: str, num_cameras: int):
    image_dir = os.path.join(data_dir, split)
    images = []
    cameras = CameraInfo.from_json(os.path.join(data_dir, "{}_cameras.json".format(split)))
    cameras = cameras[:num_cameras]
    for camera in cameras:
        image = cv2.imread(os.path.join(image_dir, camera.name + ".png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        images.append(image)

    images = np.stack(images)
    return cameras, images


def _parse_args():
    parser = argparse.ArgumentParser("Tiny NeRF")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("voxels_dir", help="Path to the voxels directory")
    parser.add_argument("results_dir", help="Path to output results")
    parser.add_argument("--path-length", type=int, default=128,
                        help="Number of voxels to intersect")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=200,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=100,
                        help="Number of cameras")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gauss-sigma", type=float, default=6.05,
                        help="Value of sigma for the gaussian model")
    return parser.parse_args()


def _main():
    args = _parse_args()
    data = np.load(os.path.join(args.voxels_dir, "carving.npz"))
    voxels = OcTree.load(data)
    opacity = data["opacity"]

    model = GaussianFourierMLP(3, 4, args.gauss_sigma)
    initial_output = torch.FloatTensor([1e-5, 0.5, 0.5, 1e-5])
    model.output.bias.data = torch.logit(initial_output)

    train_cameras, train_images = _load_images(args.data_dir, "train",
                                               args.num_cameras)
    train_dataset = RaySamplingDataset(train_images, train_cameras, voxels,
                                       args.path_length, args.num_samples,
                                       args.resolution, opacity)

    val_cameras, val_images = _load_images(args.data_dir, "val",
                                           args.num_cameras)
    val_dataset = RaySamplingDataset(val_images, val_cameras, voxels,
                                     args.path_length, args.num_samples,
                                     args.resolution, opacity)

    trainer = TinyNeRF(train_dataset, val_dataset, model, args.results_dir)
    trainer.to("cuda")

    trainer.fit(1024, args.learning_rate, 10)


if __name__ == "__main__":
    _main()
