import argparse
import os
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
    def __init__(self, cameras: List[CameraInfo], images: np.ndarray, num_levels: int, max_voxels: int):
        nn.Module.__init__(self)
        self.voxels = OcTree(num_levels)
        self._max_voxels = max_voxels
        
        positions = self.voxels.leaf_centers()
        positions = torch.from_numpy(positions)
        self.positions = nn.Parameter(positions, requires_grad=False)

        images = torch.from_numpy(images)
        self.images = nn.Parameter(images, requires_grad=False)

        self.cameras = cameras

        opacity = torch.zeros(len(positions), dtype=torch.float32)
        self.opacity = nn.Parameter(opacity)

        self.log = []

    def _update_rays(self):
        positions = self.positions.cpu().numpy()
        leaves = []
        deltas = []
        points = []
        max_dist = torch.zeros((len(positions), 1), dtype=torch.float32)
        max_dist[:] = 1e10
        for camera in self.cameras:
            cam_points = camera.project(positions)
            starts, dirs = camera.raycast(cam_points)
            voxels = self.voxels.batch_intersect(starts, dirs)
            cam_deltas = voxels.t_stops[:, 1:] - voxels.t_stops[:, :-1]
            cam_deltas = torch.cat([cam_deltas, max_dist], dim=-1)
            deltas.append(cam_deltas)
            leaves.append(voxels.leaf_index)
            points.append(torch.from_numpy(cam_points))

        device = self.images.device
        self.leaves = [torch.from_numpy(x).to(device) for x in leaves]
        self.deltas = [torch.from_numpy(d).to(device) for d in deltas]
        points = torch.stack(points)
        points = points.unsqueeze(1)
        images = self.images.unsqueeze(1)
        self.targets = F.grid_sample(images, points, align_corners=False, padding_mode="zeros")

    def forward(self) -> torch.Tensor:
        # for camera in cameras
        # 1. sample the leaf opacity values
        # 2. compute the volume equation
        outputs = []
        left_trans = torch.ones((len(self.positions), 1), dtype=torch.float32, device=self.opacity.device)
        for leaves, deltas in zip(self.leaves, self.deltas):
            opacity = self.opacity[leaves]
            alpha = 1 - torch.exp(-(opacity * deltas))
            trans = torch.minimum(1, 1 - alpha + 1e-10)
            _, trans = trans.split([1, leaves.shape[1] - 1])
            trans = torch.cat([left_trans, trans], -1)
            weights = alpha * torch.cumprod(trans, -1)
            cam_output = weights.sum(-1)
            outputs.append(cam_output)

        return torch.stack(outputs)

    def _loss(self, images: torch.Tensor):
        targets, weights = self.forward(images)
        opacity = self.opacity.clamp(0, 1)
        opacity = opacity.unsqueeze(0).expand(targets.shape[0], -1)
        loss = (targets - opacity).square() * weights
        loss = loss.sum()
        return loss

    def _log(self, iteration, loss):
        opacity = self.opacity.clamp(0, 1).detach().cpu().numpy()
        occupied = (opacity * 5).astype(np.int32)
        centers = self.voxels.leaf_centers()
        scales = self.voxels.leaf_scales()
        print(iteration, loss, np.bincount(occupied, minlength=5).tolist())
        self.log.append((
            iteration,
            self.camera_transform.rotation_quat.detach().cpu().numpy(),
            self.camera_transform.translate_vec.detach().cpu().numpy(),
            opacity,
            centers,
            scales))

    def fit(self, images: torch.Tensor, num_steps: int, num_finetune: int):
        opacity_params = {
            "params": [self.opacity],
            "lr": 1e-2
        }

        optim = torch.optim.Adam([opacity_params])
        
        def _closure():
            optim.zero_grad()
            loss = self._loss(images)
            loss.backward()
            return loss

        add_camera = num_steps // 4
        cameras_added = False
        for i in range(num_steps):
            optim.step(_closure)
            if i >= add_camera and not cameras_added:
                print("Starting to optimize the cameras...")
                optim = torch.optim.Adam([opacity_params, *self.camera_transform.param_groups])
                cameras_added = True
                
            if i % 10 == 0:
                self._log(i, self._loss(images).item())

        print("Merging octree...")
        opacity = self.opacity.detach().cpu().numpy()
        for i in range(4):
            opacity = self.voxels.merge(opacity, 0.1)

        print("Splitting octree...")
        for i in range(4):
            opacity = self.voxels.split(opacity, 0.1)

        positions = self.voxels.leaf_centers()
        positions = torch.from_numpy(positions).to(device=images.device)
        opacity = torch.from_numpy(opacity).to(device=images.device)
        self.opacity = nn.Parameter(opacity)
        self.positions = nn.Parameter(positions, requires_grad=False)
        opacity_params["params"] = [self.opacity]

        print("Finetuning opacity...")
        optim = torch.optim.Adam([opacity_params])
        for i in range(num_steps, num_steps + num_finetune):
            optim.step(_closure)
            if i % 10 == 0:
                self._log(i, self._loss(images).item())

        self._log(num_steps + num_finetune, self._loss(images).item())


def _extract_masks(image_dir):
    masks_dir = os.path.join(image_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    extractor = None
    for name in os.listdir(image_dir):
        if not name.endswith(".png"):
            continue

        mask_path = os.path.join(masks_dir, name)
        if os.path.exists(mask_path):
            continue

        image = cv2.imread(os.path.join(image_dir,name))

        chroma = [255, 255, 255]
        test = (image == chroma).sum(-1)
        mask = test == 3
        cv2.imwrite(mask_path, mask)


def _carve_and_calibrate(image_dir, initial_depth, max_voxels, num_steps, num_finetune):
    mask_dir = os.path.join(image_dir, "masks")
    device = "cuda"

    images = []
    for name in os.listdir(mask_dir):
        if not name.endswith(".png"):
            continue

        image = cv2.imread(os.path.join(mask_dir, name))
        images.append(image[:, :, 0] != 0)

    images = np.stack(images).astype(np.float32)
    images = torch.from_numpy(images).to(device=device)

    num_cameras, height, width = images.shape[:3]
    angle_diff = 2 * np.pi / (len(images) - 1)
    elevation = -np.pi / 6
    cameras = []
    for i in range(num_cameras):
        position = np.array([0, 0, 3, 1], np.float32)
        position = sp.Transforms.rotation_about_x(elevation) @ position
        position = sp.Transforms.rotation_about_y(i * angle_diff) @ position
        position = sp.Transforms.translate([0, -1, 0]) @ position
        position = position[:3]
        rotation = np.linalg.inv(sp.Transforms.look_at_rotation([0, 0, 0], position, [0, -1, 0]))
        camera_to_world = sp.Transforms.translate(position) @ rotation
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = width
        intrinsics[0, 2] = width / 2
        intrinsics[1, 1] = width
        intrinsics[1, 2] = height / 2
        cameras.append(CameraInfo.create("cam{}".format(i), (width, height), intrinsics, camera_to_world))

    occupancy = Occupancy(cameras, initial_depth, max_voxels)
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
    return parser.parse_args()


def _main():
    args = _parse_args()
    _extract_masks(args.image_dir)
    _carve_and_calibrate(args.image_dir, args.initial_depth, args.max_voxels, args.num_steps, args.num_finetune)
    _create_carve_scenepic(args.image_dir)


if __name__ == "__main__":
    _main()
