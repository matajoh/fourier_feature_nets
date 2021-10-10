import argparse
import os
from typing import List

import cv2
from matplotlib.pyplot import get_cmap
from nerf2d import CameraInfo, CameraTransform, OcTree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scenepic as sp
from scipy.spatial.transform import Rotation


class GrabCutExtractor:
    def __init__(self, resolution, stencil_radius=8, fg_color=(10, 255, 10), bg_color=(10, 10, 255)):
        self.original_image = None
        self.image = None
        self.stencil_radius = stencil_radius
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.mask = None
        self.mask_image = None
        self.width, self.height = resolution
        self.background = np.zeros((1, 65), np.float64)
        self.foreground = np.zeros((1, 65), np.float64)
        self.frame = np.zeros((self.height, self.width * 2, 3), np.uint8)
        self.color = None

    def draw(self, x, y):
        if self.color is not None:
            self.mask_image = cv2.circle(self.mask_image, (x, y), self.stencil_radius, self.color, -1)

    def extract_foreground(self, image, reuse_mask=False):
        self.original_image = image
        height, width = image.shape[:2]
        if width != self.width or height != self.height:
            sigma = width / self.width
            self.image = cv2.GaussianBlur(image, (0, 0), sigma)
            self.image = cv2.resize(image, (self.width, self.height))
        else:
            self.image = image

        rect = None
        init_flag = cv2.GC_INIT_WITH_MASK
        if not reuse_mask or self.mask is None:
            self.mask = np.zeros(self.image.shape[:2], np.uint8)
            rect = (25, 5, self.width - 25, self.height - 5)
            init_flag = cv2.GC_INIT_WITH_RECT
        else:
            self.mask[self.mask == cv2.GC_BGD] = cv2.GC_PR_BGD
            self.mask[self.mask == cv2.GC_FGD] = cv2.GC_PR_FGD

        self.mask_image = np.zeros_like(self.image)
        print("\nPerforming Initial Background Subtraction")
        cv2.grabCut(self.image, self.mask, rect, self.background, self.foreground, 5, init_flag)

        cv2.namedWindow(winname="Subtract Background")
        cv2.setMouseCallback("Subtract Background", paint, self)
        while True:
            fg_mask = (self.mask_image == self.fg_color).sum(-1)
            bg_mask = (self.mask_image == self.bg_color).sum(-1)
            mask = self.mask.copy()
            mask[fg_mask == 3] = cv2.GC_FGD
            mask[bg_mask == 3] = cv2.GC_BGD

            mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
            self.frame[:, :self.width, :] = mask2[:, :, np.newaxis] * self.image
            masked_image = mask2[:, :, np.newaxis] * 63
            self.frame[:, self.width:, :] = np.where(self.mask_image != 0, self.mask_image, masked_image)
            cv2.imshow("Subtract Background", self.frame)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break

            if key & 0xFF == 32:
                print("\nPerforming Background Subtraction")
                
                cv2.grabCut(self.image, mask, None, self.background, self.foreground, 5, cv2.GC_INIT_WITH_MASK)
                self.mask = mask
                self.mask_image[:] = 0
                print("\nExtraction complete")

        cv2.destroyAllWindows()
        mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
        if width != self.width or height != self.height:
            mask2 = cv2.resize(mask2, (width, height))

        return mask2[:, :, np.newaxis] * self.original_image


class Occupancy(nn.Module):
    def __init__(self, cameras: List[CameraInfo], num_levels, max_voxels):
        nn.Module.__init__(self)
        self.voxels = OcTree(num_levels, max_voxels)
        
        positions = self.voxels.leaf_centers()
        positions = torch.from_numpy(positions)
        self.positions = nn.Parameter(positions, requires_grad=False)

        opacity = torch.zeros(len(positions), dtype=torch.float32)
        self.opacity = nn.Parameter(opacity)

        self.camera_transform = CameraTransform(cameras)

        self.log = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        points, depth = self.camera_transform(self.positions, True)
        points = points.unsqueeze(1)
        images = images.unsqueeze(1)
        targets = F.grid_sample(images, points, align_corners=False)
        targets = targets.reshape(points.shape[0], -1)
        weights = torch.exp(-depth)
        return targets, weights

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

    def fit(self, images: torch.Tensor, num_steps=1500, merge_split_freq=100):
        opacity_params = {
            "params": [self.opacity],
            "lr": 1e-2
        }

        camera_params = {
            "params": [self.camera_transform.rotation_quat, self.camera_transform.translate_vec],
            "lr": 1e-5
        }

        optim = torch.optim.Adam([opacity_params, camera_params])
        
        def _closure():
            optim.zero_grad()
            loss = self._loss(images)
            loss.backward()
            return loss

        merge_start = merge_split_freq + merge_split_freq // 2
        merge_stop = num_steps // 2
        for i in range(num_steps):
            optim.step(_closure)
            if merge_start <= i < merge_stop and (i - merge_start) % merge_split_freq == 0:
                print("Merging/Splitting...")
                opacity = self.opacity.detach().cpu().numpy()
                opacity = self.voxels.merge_and_split(opacity, 0.1)
                positions = self.voxels.leaf_centers()
                positions = torch.from_numpy(positions).to(device=images.device)
                opacity = torch.from_numpy(opacity).to(device=images.device)
                self.opacity = nn.Parameter(opacity)
                self.positions = nn.Parameter(positions, requires_grad=False)
                opacity_params["params"] = [self.opacity]
                optim = torch.optim.Adam([opacity_params, camera_params])
                
            if i % 10 == 0:
                self._log(i, self._loss(images).item())

        self._log(num_steps, self._loss(images).item())

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
        
        if extractor is None:
            extractor = GrabCutExtractor((image.shape[1], image.shape[0]))

        mask = extractor.extract_foreground(image, True)
        cv2.imwrite(mask_path, mask)


def _carve_and_calibrate(image_dir, initial_depth, max_voxels, num_steps, merge_split_freq):
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
    angle_diff = 2 * np.pi / len(images)
    elevation = -np.pi / 6
    cameras = []
    for i in range(num_cameras):
        position = np.array([0, 0, 3, 1], np.float32)
        position = sp.Transforms.rotation_about_x(elevation) @ position
        position = sp.Transforms.rotation_about_y(i * angle_diff) @ position
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

    occupancy.fit(images, num_steps, merge_split_freq)
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


def _create_carve_scenepic(image_dir):
    mask_dir = os.path.join(image_dir, "masks")
    images = []
    for name in os.listdir(mask_dir):
        if not name.endswith(".png"):
            continue

        image = cv2.imread(os.path.join(mask_dir, name))
        images.append(image[:, :, 0] != 0)

    height, width = images[0].shape[:2]
    data = np.load("carving.npz")

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

        frustums.add_camera_frustum(camera, colors[i], depth=0.2, thickness=0.01)

    centers = data["centers1500"]
    scales = data["scales1500"]
    opacity = data["opacity1500"]
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
    parser.add_argument("--merge-split-freq", type=int, default=100, help="How often to merge/split the voxel octree")
    return parser.parse_args()


def _main():
    args = _parse_args()
    _extract_masks(args.image_dir)
    _carve_and_calibrate(args.image_dir, args.initial_depth, args.max_voxels, args.num_steps, args.merge_split_freq)
    _create_carve_scenepic(args.image_dir)


if __name__ == "__main__":
    _main()
