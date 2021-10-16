import argparse
from collections import namedtuple
import json
import math
import os
import sys
from typing import List

import cv2
from matplotlib.pyplot import get_cmap
from nerf import CameraInfo, OcTree, VoxelDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import scenepic as sp


class VolumeCarver(nn.Module):
    def __init__(self, depth: int, scale: float, path_length: int,
                 image_dir: str, blob_prior_weight: float, resolution: int):
        nn.Module.__init__(self)
        self.voxels = OcTree(depth, scale)
        print(depth, len(self.voxels))
        self._path_length = path_length
        self._image_dir = image_dir
        self._blob_prior_weight = blob_prior_weight
        self._resolution = resolution
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        logits = torch.full((self.voxels.num_leaves,), -10.0, dtype=torch.float32)
        self.logits = nn.Parameter(logits)

        self._blob_prior = nn.Parameter(self._compute_distance_weights())

        self.log = []

    def _compute_distance_weights(self) -> torch.Tensor:
        positions = torch.from_numpy(self.voxels.leaf_centers())
        squared_dist = positions.square().sum(-1)
        squared_dist /= self.voxels.scale * self.voxels.scale * 3
        return squared_dist

    def forward(self, t_stops: torch.Tensor,
                leaves: torch.Tensor) -> torch.Tensor:
        num_rays, length = leaves.shape
        max_dist = torch.full((num_rays, 1), 1e10, dtype=torch.float32,
                              device=self.logits.device)
        left_trans = torch.ones_like(max_dist)
        opacity = torch.sigmoid(self.logits)
        opacity = opacity[leaves]
        deltas = t_stops[:, 1:] - t_stops[:, :-1]
        deltas = torch.cat([deltas, max_dist], axis=-1)
        alpha = 1 - torch.exp(-(opacity * deltas))
        ones = torch.ones_like(alpha)
        trans = torch.minimum(ones, 1 - alpha + 1e-10)
        _, trans = trans.split([1, length - 1], dim=-1)
        trans = torch.cat([left_trans, trans], -1)
        weights = alpha * torch.cumprod(trans, -1)
        output = weights.sum(-1)
        return output

    def _val_image(self, camera: CameraInfo) -> np.ndarray:
        x_vals = np.linspace(-1, 1, self._resolution)
        y_vals = np.linspace(-1, 1, self._resolution)
        points = np.stack(np.meshgrid(x_vals, y_vals), -1).reshape(-1, 2)
        starts, directions = camera.raycast(points)
        t_stops, leaves = self.voxels.intersect(starts, directions,
                                                self._path_length)
        t_stops = torch.from_numpy(t_stops)
        leaves = torch.from_numpy(leaves)
        pixels = self.forward(t_stops, leaves)
        pixels = pixels.detach().cpu().numpy()
        pixels = pixels.reshape(self._resolution, self._resolution)
        pixels = (pixels * 255).astype(np.uint8)
        return pixels

    def _loss(self, t_stops: torch.Tensor, leaves: torch.Tensor,
              targets: torch.Tensor, debug: bool) -> torch.Tensor:
        outputs = self.forward(t_stops, leaves)
        render_energy = (targets - outputs).square().mean()
        opacity = torch.sigmoid(self.logits)
        blob_energy = (self._blob_prior * opacity).square().sum()
        loss = render_energy + self._blob_prior_weight * blob_energy
        if debug:
            return namedtuple("Loss", ["render", "blob"])(
                render_energy.item(),
                blob_energy.item()
            )

        return loss

    @property
    def opacity(self) -> np.ndarray:
        return torch.sigmoid(self.logits).detach().cpu().numpy()

    def _log(self, iteration, loss):
        occupied = (self.opacity * 5).astype(np.int32)
        print(iteration, loss,
              np.bincount(occupied, minlength=5).tolist())

    def fit(self, images: torch.Tensor, cameras: List[CameraInfo],
            batch_size: int, num_epochs: int, learning_rate=1.0):
        optim = torch.optim.Adam([self.logits], learning_rate)

        dataset = VoxelDataset(images, cameras, self.voxels,
                               self._path_length, self._resolution)
        data_loader = DataLoader(dataset, batch_size, True)

        device = self.logits.device
        viz_camera = 0
        for epoch in range(num_epochs):
            print("Epoch", epoch)
            for step, (t_stops, leaves, targets) in enumerate(data_loader):
                t_stops = t_stops.to(device)
                leaves = leaves.to(device)
                targets = targets.to(device)

                optim.zero_grad()
                loss = self._loss(t_stops, leaves, targets, False)
                loss.backward()
                optim.step()

                if step % 5 == 0:
                    with torch.no_grad():
                        self._log(step, self._loss(t_stops, leaves, targets, True))

                    with torch.no_grad():
                        name = "e{:04}_b{:03}_c{:03}.png".format(epoch, step, viz_camera)
                        path = os.path.join(self._image_dir, name)
                        cv2.imwrite(path, self._val_image(cameras[viz_camera]))
                        viz_camera = (viz_camera + 1) % len(cameras)


def _convert_cameras(image_dir, split, scale=1):
    cameras: List[CameraInfo] = []
    images: List[np.ndarray] = []
    cameras_path = os.path.join(image_dir, "transforms_{}.json".format(split))
    with open(cameras_path) as file:
        data = json.load(file)
        fov_x = data["camera_angle_x"]

        for frame in data["frames"]:
            file_path = frame["file_path"]
            name = file_path[file_path.rindex("/") + 1:]
            image_path = os.path.join(image_dir, file_path + ".png")
            image = cv2.imread(image_path)
            image = image[:, :, ::-1]
            images.append(cv2.resize(image, (200, 200), cv2.INTER_AREA))
            height, width, _ = image.shape
            if scale != 1:
                height = int(scale * height)
                width = int(scale * width)

            aspect_ratio = width / height
            focal_length = .5 * width / math.tan(.5 * fov_x)
            intrinsics = np.eye(3, dtype=np.float32)
            intrinsics[0, 0] = focal_length
            intrinsics[0, 2] = width / 2
            intrinsics[1, 1] = focal_length * aspect_ratio
            intrinsics[1, 2] = height / 2
            camera_to_world = np.array(frame["transform_matrix"], np.float32)
            camera_to_world = camera_to_world @ sp.Transforms.rotation_about_x(np.pi)
            cameras.append(CameraInfo.create(name,
                                             (width, height),
                                             intrinsics,
                                             camera_to_world))

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=800, height=800)
    cube_mesh = scene.create_mesh("cube", shared_color=sp.Colors.Blue)
    cube_mesh.add_cube(transform=sp.Transforms.scale(4))
    frustums = scene.create_mesh("frustums", layer_id="frustums", shared_color=sp.Colors.White)
    image_meshes = []
    camera_labels = []
    for camera, pixels in zip(cameras, images):
        sp_camera = camera.to_scenepic()
        frustums.add_camera_frustum(sp_camera, depth=0.5, thickness=0.01)
        label = scene.create_label(text=camera.name)
        camera_labels.append(label)

        image = scene.create_image()
        image.from_numpy(pixels)
        image_mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id, double_sided=True)
        image_mesh.add_camera_image(sp_camera, depth=0.5)
        image_meshes.append(image_mesh)

    for camera, label in zip(cameras, camera_labels):
        sp_camera = camera.to_scenepic()

        frame = canvas.create_frame()
        frame.camera = sp_camera
        frame.add_mesh(cube_mesh)
        frame.add_label(label, camera.position)
        frame.add_mesh(frustums)
        for image_mesh in image_meshes:
            frame.add_mesh(image_mesh)

    scene.save_as_html(os.path.join(image_dir, "{}_cameras.html".format(split)))
    CameraInfo.to_json(os.path.join(image_dir, "{}_cameras.json".format(split)), cameras)


def _extract_masks(image_dir, scale=1):
    masks_dir = os.path.join(image_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    for name in os.listdir(image_dir):
        if not name.endswith(".png"):
            continue

        mask_path = os.path.join(masks_dir, name)
        image = cv2.imread(os.path.join(image_dir, name))

        height, width, _ = image.shape

        if scale != 1:
            height = int(scale * height)
            width = int(scale * width)
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)

        chroma = [0, 0, 0]
        test = (image == chroma).sum(-1)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[test != 3] = 255
        cv2.imwrite(mask_path, mask)


def _carve(params):
    mask_dir = os.path.join(params["image_dir"], "train", "masks")
    device = "cpu"

    cameras = CameraInfo.from_json(os.path.join(params["image_dir"], "train_cameras.json"))
    cameras = cameras[:params["num_cameras"]]
    images = []
    for camera in cameras:
        image_path = os.path.join(mask_dir, camera.name + ".png")
        image = cv2.imread(image_path)
        images.append(image[:, :, 0] != 0)

    images = np.stack(images).astype(np.float32)

    occupancy = VolumeCarver(params["depth"],
                             params["scale"],
                             params["path_length"],
                             params["results_dir"],
                             params["blob_prior_weight"],
                             params["resolution"])
    occupancy.to(device)

    occupancy.fit(images, cameras, 50000, params["num_epochs"])
    entries = occupancy.voxels.state_dict
    entries["opacity"] = occupancy.opacity
    entries["names"] = [camera.name for camera in cameras]
    entries["intrinsics"] = np.stack([camera.intrinsics for camera in cameras])
    entries["extrinsics"] = np.stack([camera.extrinsics for camera in cameras])

    np.savez(os.path.join(params["results_dir"], "carving.npz"), **entries)


def _create_carve_scenepic(image_dir: str, results_dir: str):
    print("Creating scenepic")
    mask_dir = os.path.join(image_dir, "train", "masks")
    data = np.load(os.path.join(results_dir, "carving.npz"))
    print("Loading images...")
    images = []
    for name in data["names"]:
        image_path = os.path.join(mask_dir, name + ".png")
        image = cv2.imread(image_path)
        images.append(image[:, :, 0] != 0)

    height, width = images[0].shape[:2]

    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    num_cameras = len(intrinsics)
    voxels = OcTree.load(data)

    scene = sp.Scene()
    frustums = scene.create_mesh("frustums", layer_id="frustums")
    canvas = scene.create_canvas_3d(width=width * 2, height=height * 2)
    cmap = get_cmap("jet")
    colors = cmap(np.linspace(0, 1, num_cameras))[:, :3]
    image_meshes = []
    cameras = []
    sys.stdout.write("Adding cameras")
    for i in range(num_cameras):
        sys.stdout.write(".")
        sys.stdout.flush()
        gl_world_to_camera = sp.Transforms.gl_world_to_camera(extrinsics[i])
        gl_projection = sp.Transforms.gl_projection(intrinsics[i], width, height, 0.1, 100)
        camera = sp.Camera(world_to_camera=gl_world_to_camera, projection=gl_projection)
        cameras.append(camera)

        image = scene.create_image("cam_image{}".format(i))
        image.from_numpy(images[i])
        mesh = scene.create_mesh("cam_image{}".format(i), layer_id="images",
                                 texture_id=image.image_id, double_sided=True)
        mesh.add_camera_image(camera, depth=0.5)
        image_meshes.append(mesh)

        frustums.add_camera_frustum(camera, colors[i], depth=0.5, thickness=0.01)

    print("done.")

    sys.stdout.write("Adding voxels")
    centers = voxels.leaf_centers()
    scales = voxels.leaf_scales()
    opacity = data["opacity"]
    num_bands = 8
    colors = cmap(np.linspace(0, 1, num_bands))[:, :3]
    # we want five meshes to represent the different levels
    voxel_meshes = [scene.create_mesh("voxels{}".format(i),
                                      layer_id="{:.2f}->{:.2f}".format(i*1.0/num_bands, (i+1)*1.0/num_bands),
                                      shared_color=colors[i])
                    for i in range(num_bands)]

    positions = [[] for _ in range(num_bands)]
    opacity = np.minimum(opacity, 0.99) * num_bands
    opacity = opacity.astype(np.int32)
    for i, (center, index) in enumerate(zip(centers, opacity)):
        if i % 10000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        positions[index].append(center)

    for mesh, band_positions in zip(voxel_meshes, positions):
        positions = np.stack(band_positions)
        mesh.add_cube(transform=sp.Transforms.scale(scales[0] * 2))
        mesh.enable_instancing(positions)

    print("done.")

    print("Creating frames")
    for i in range(num_cameras):
        frame = canvas.create_frame(camera=cameras[i])
        frame.add_mesh(frustums)
        for mesh in voxel_meshes:
            frame.add_mesh(mesh)

        for mesh in image_meshes:
            frame.add_mesh(mesh)

    scene.framerate = 10
    scene.save_as_html(os.path.join(results_dir, "carve.html"))


def _parse_args():
    parser = argparse.ArgumentParser("Volume Dataset Preparation")
    parser.add_argument("image_dir", help="Image containing the training images")
    parser.add_argument("--num-cameras", type=int, default=10, help="Number of cameras to use")
    parser.add_argument("--depth", type=int, default=5, help="Depth of the voxel octree")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of carving epochs")
    parser.add_argument("--scale", type=float, default=2, help="Scale of voxel space")
    parser.add_argument("--path-length", type=int, default=50, help="Number of voxels to intersect")
    parser.add_argument("--results-dir", default="voxels", help="Output directory for results")
    parser.add_argument("--blob-prior-weight", type=float, default=1e-3, help="Weight for the blob prior")
    parser.add_argument("--resolution", type=int, default=200, help="Image size to use during training")
    return parser.parse_args()


def _main():
    args = _parse_args()
    #for split in ["train", "val", "test"]:
    #    _convert_cameras(args.image_dir, split, 0.5)

    #_extract_masks(os.path.join(args.image_dir, "train"), 0.5)
    _carve(vars(args))
    _create_carve_scenepic(args.image_dir, args.results_dir)


if __name__ == "__main__":
    _main()
