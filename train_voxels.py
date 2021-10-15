import argparse
import json
import math
import os
import timeit
from typing import List

import cv2
from matplotlib.pyplot import get_cmap
from nerf import CameraInfo, OcTree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scenepic as sp


class Occupancy(nn.Module):
    def __init__(self, cameras: List[CameraInfo], images: np.ndarray, num_levels: int, max_voxels: int, scale: float):
        nn.Module.__init__(self)
        self.voxels = OcTree(num_levels, scale)
        self._max_voxels = max_voxels
        
        positions = self.voxels.leaf_centers()
        positions = torch.from_numpy(positions)
        self.positions = nn.Parameter(positions, requires_grad=False)

        images = torch.from_numpy(images)
        self.images = nn.Parameter(images, requires_grad=False)

        self.cameras = cameras
        self.num_cameras = len(cameras)

        opacity = torch.zeros(len(positions), dtype=torch.float32)
        self.opacity = nn.Parameter(opacity)

        self._update_rays()

        self.log = []

    def _update_rays(self):
        positions = self.positions.cpu().numpy()
        leaves = []
        deltas = []
        points = []
        max_dist = np.zeros((1, len(positions)), dtype=np.float32)
        max_dist[:] = 1e10
        print("Casting rays...")
        for camera in self.cameras:
            print(camera.name)
            cam_points = camera.project(positions)
            starts, dirs = camera.raycast(cam_points)
            time = timeit.timeit(lambda: self.voxels.intersect(starts, dirs), number=3)
            print("Best of 3:", time, "s")
            #voxels = self.voxels.batch_intersect(starts, dirs)
            #cam_deltas = voxels.t_stops[1:] - voxels.t_stops[:-1]
            #cam_deltas = np.concatenate([cam_deltas, max_dist])
            #deltas.append(cam_deltas.T)
            #leaves.append(voxels.leaf_index.T)
            #points.append(torch.from_numpy(cam_points))

        device = self.images.device
        leaves = [torch.from_numpy(x).to(device) for x in leaves]
        self.leaves = nn.ParameterList([nn.Parameter(x, requires_grad=False) for x in leaves])
        deltas = [torch.from_numpy(d).to(device) for d in deltas]
        self.deltas = nn.ParameterList([nn.Parameter(d, requires_grad=False) for d in deltas])
        points = torch.cat(points)
        points = points.unsqueeze(1)
        images = self.images.unsqueeze(1)
        targets = F.grid_sample(images, points, align_corners=False, padding_mode="zeros")
        self.targets = nn.Parameter(targets, requires_grad=False)

    def forward(self, camera: int) -> torch.Tensor:
        left_trans = torch.ones((len(self.positions), 1), dtype=torch.float32, device=self.opacity.device)
        leaves = self.leaves[camera]
        deltas = self.deltas[camera]
        opacity = self.opacity[leaves]
        alpha = 1 - torch.exp(-(opacity * deltas))
        ones = torch.ones_like(alpha)
        trans = torch.minimum(ones, 1 - alpha + 1e-10)
        _, trans = trans.split([1, leaves.shape[1] - 1], dim=-1)
        trans = torch.cat([left_trans, trans], -1)
        weights = alpha * torch.cumprod(trans, -1)
        output = weights.sum(-1)
        return output

    def _loss(self, camera: int):
        outputs = self.forward(camera)
        loss = (self.targets[camera] - outputs).square().sum()
        return loss

    def _log(self, iteration, loss):
        opacity = self.opacity.clamp(0, 1).detach().cpu().numpy()
        occupied = (opacity * 5).astype(np.int32)
        centers = self.voxels.leaf_centers()
        scales = self.voxels.leaf_scales()
        print(iteration, loss, np.bincount(occupied, minlength=5).tolist())
        self.log.append((
            iteration,
            opacity,
            centers,
            scales))

    def fit(self, images: torch.Tensor, num_steps: int, num_finetune: int):
        optim = torch.optim.Adam([self.opacity], 3e-5)

        # TODO implement minibatching with rays
        # TODO speed up ray casting
        # 1. create ray dataset
        # 2. sample dataset
        
        def _closure():
            optim.zero_grad()
            loss = 0
            for i in range(self.num_cameras):
                loss += self._loss(i)

            loss.backward()
            return loss

        for i in range(num_steps):
            optim.step(_closure)             
            if i % 10 == 0:
                self._log(i, self._loss(0).item())

        print("Merging octree...")
        opacity = self.opacity.detach().cpu().numpy()
        for _ in range(4):
            opacity = self.voxels.merge(opacity, 0.1)

        print("Splitting octree...")
        for _ in range(4):
            opacity = self.voxels.split(opacity, 0.1)

        positions = self.voxels.leaf_centers()
        positions = torch.from_numpy(positions).to(device=images.device)
        opacity = torch.from_numpy(opacity).to(device=images.device)
        self.opacity = nn.Parameter(opacity)
        self.positions = nn.Parameter(positions, requires_grad=False)
        self._update_rays()

        print("Finetuning opacity...")
        optim = torch.optim.Adam([self.opacity], 1e-3)
        for i in range(num_steps, num_steps + num_finetune):
            optim.step(_closure)
            if i % 10 == 0:
                self._log(i, self._loss(0).item())

        self._log(num_steps + num_finetune, self._loss(0).item())


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


def _carve_and_calibrate(image_dir, initial_depth, max_voxels, num_steps, num_finetune, scale):
    mask_dir = os.path.join(image_dir, "train", "masks")
    device = "cuda"

    cameras = CameraInfo.from_json(os.path.join(image_dir, "train_cameras.json"))
    cameras = cameras[:10]
    images = []
    for camera in cameras:
        image_path = os.path.join(mask_dir, camera.name + ".png")
        image = cv2.imread(image_path)
        images.append(image[:, :, 0] != 0)

    images = np.stack(images).astype(np.float32)

    occupancy = Occupancy(cameras, images, initial_depth, max_voxels, scale)
    occupancy.to(device)

    occupancy.fit(images, num_steps, num_finetune)
    entries = {}
    entries["num_entries"] = len(occupancy.log)
    entries["iterations"] = np.stack([entry[0] for entry in occupancy.log])
    entries["intrinsics"] = occupancy.camera_transform.intrinsics
    entries["rotation_quats"] = np.stack([entry[1] for entry in occupancy.log])
    entries["translate_vecs"] = np.stack([entry[2] for entry in occupancy.log])
    for i, _, _, opacity, centers, scales in occupancy.log:
        entries["opacity{:04}".format(i)] = opacity
        entries["centers{:04}".format(i)] = centers
        entries["scales{:04}".format(i)] = scales

    np.savez(os.path.join(image_dir, "carving.npz"), **entries)
    occupancy.voxels.save(os.path.join(image_dir, "voxels.npz"))

    cameras_path = os.path.join(image_dir, "cameras.json")
    CameraInfo.to_json(cameras_path, occupancy.camera_transform.camera_info)


def _create_carve_scenepic(image_dir):
    mask_dir = os.path.join(image_dir, "masks")
    images = []
    for name in os.listdir(mask_dir):
        if not name.endswith(".png"):
            continue

        image = cv2.imread(os.path.join(mask_dir, name))
        images.append(image[:, :, 0] != 0)

    height, width = images[0].shape[:2]
    data = np.load(os.path.join(image_dir, "carving.npz"))

    intrinsics = data["intrinsics"]
    rotations = [Rotation.from_quat(rquat).as_matrix() for rquat in data["rotation_quats"][-1]]
    translations = [sp.Transforms.translate(tvec) for tvec in data["translate_vecs"][-1]]
    num_cameras = len(intrinsics)

    scene = sp.Scene()
    frustums = scene.create_mesh("frustums", layer_id="frustums")
    canvas = scene.create_canvas_3d(width=width, height=height)
    cmap = get_cmap("jet")
    colors = cmap(np.linspace(0, 1, num_cameras))[:, :3]
    image_meshes = []
    cameras = []
    for i in range(num_cameras):
        world_to_camera = translations[i].copy()
        world_to_camera[:3, :3] = rotations[i]
        camera_to_world = np.linalg.inv(world_to_camera)

        gl_world_to_camera = sp.Transforms.gl_world_to_camera(camera_to_world)
        gl_projection = sp.Transforms.gl_projection(intrinsics[i], width, height, 0.1, 100)
        camera = sp.Camera(world_to_camera=gl_world_to_camera, projection=gl_projection)
        cameras.append(camera)

        image = scene.create_image("cam_image{}".format(i))
        image.from_numpy(images[i])
        mesh = scene.create_mesh("cam_image{}".format(i), layer_id="images", texture_id=image.image_id, double_sided=True)
        mesh.add_camera_image(camera, depth=0.2)
        image_meshes.append(mesh)

        frustums.add_camera_frustum(camera, colors[i], depth=0.2, thickness=0.004)

    final = data["iterations"][-1]
    centers = data["centers{:04}".format(final)]
    scales = data["scales{:04}".format(final)]
    opacity = data["opacity{:04}".format(final)]
    colors = cmap(np.linspace(0, 1, 5))[:, :3]
    # we want five meshes to represent the different levels
    voxel_meshes = [scene.create_mesh("voxels{}".format(i),
                                      layer_id="{:.2f}->{:.2f}".format(i*0.2, (i+1)*0.2),
                                      shared_color=colors[i])
                    for i in range(5)]
    for center, scale, opacity in zip(centers, scales, opacity):
        size = pow(2.0, 1-scale)
        index = int(min(opacity, 0.99) * 5)
        voxel_meshes[index].add_cube(transform=sp.Transforms.translate(center) @ sp.Transforms.scale(size))

    for i in range(num_cameras):
        frame = canvas.create_frame(camera=cameras[i])
        frame.add_mesh(frustums)
        for mesh in voxel_meshes:
            frame.add_mesh(mesh)

        for mesh in image_meshes:
            frame.add_mesh(mesh)

    frame.add_mesh(frustums)
    scene.save_as_html(os.path.join(image_dir, "carve.html"))


def _parse_args():
    parser = argparse.ArgumentParser("Volume Dataset Preparation")
    parser.add_argument("image_dir", help="Image containing the training images")
    parser.add_argument("--initial-depth", type=int, default=6, help="Initial depth of the voxel octree")
    parser.add_argument("--max-voxels", type=int, default=52000, help="Maximum number of voxels")
    parser.add_argument("--num-steps", type=int, default=1500, help="Number of carving steps")
    parser.add_argument("--num-finetune", type=int, default=500, help="Number of fine-tuning steps")
    parser.add_argument("--scale", type=float, default=2, help="Scale of voxel space")
    return parser.parse_args()


def _main():
    args = _parse_args()
    #for split in ["train", "val", "test"]:
    #    _convert_cameras(args.image_dir, split, 0.5)

    #_extract_masks(os.path.join(args.image_dir, "train"), 0.5)
    _carve_and_calibrate(args.image_dir, args.initial_depth, args.max_voxels, args.num_steps, args.num_finetune, args.scale)
    #_create_carve_scenepic(args.image_dir)


if __name__ == "__main__":
    _main()
