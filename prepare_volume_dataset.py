from collections import namedtuple
from itertools import product
import os
from typing import List

import cv2
from matplotlib.pyplot import get_cmap
from scenepic import scene
from nerf2d import CameraInfo, CameraTransform
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


OctNode = namedtuple("OctNode", ["scale", "center", "is_leaf"])

class OctTree:
    def __init__(self, depth: int):
        self.depth = depth
        self.nodes = [OctNode(0, np.array([0, 0, 0], np.float32), depth == 1)]
        num_nodes = sum([pow(8, level) for level in range(depth)])
        corners = np.array(list(product([-1, 1], [-1, 1], [-1, 1])), np.float32)

        for i in range(1, num_nodes):
            parent = (i - 1) // 8
            child = i - parent * 8 - 1
            parent_node = self.nodes[parent]
            scale = parent_node.scale + 1
            offset = pow(2, -scale) * corners[child]
            center = parent_node.center + offset
            self.nodes.append(OctNode(scale, center, scale == depth - 1))      

    def leaf_centers(self):
        return np.stack([node.center for node in self.nodes
                         if node.is_leaf])


class Occupancy(nn.Module):
    def __init__(self, cameras: List[CameraInfo], num_levels=6):
        nn.Module.__init__(self)
        self.voxels = OctTree(num_levels)
        
        positions = self.voxels.leaf_centers()
        positions = torch.from_numpy(positions)
        self.positions = nn.Parameter(positions, requires_grad=False)

        opacity = torch.zeros(len(positions), dtype=torch.float32)
        self.opacity = nn.Parameter(opacity)

        self.camera_transform = CameraTransform(cameras)

        self.log = []

    def forward(self, images: torch.Tensor):
        points = self.camera_transform(self.positions, True)
        points = points.unsqueeze(1)
        images = images.unsqueeze(1)
        targets = F.grid_sample(images, points, align_corners=False)
        targets = targets.reshape(points.shape[0], -1)
        return targets

    def _loss(self, images: torch.Tensor):
        targets = self.forward(images)       
        opacity = self.opacity.unsqueeze(0).expand(targets.shape[0], -1)
        loss = (targets - opacity).square().sum()
        return loss

    def fit(self, images: torch.Tensor, num_steps=1000):
        params = [
            self.opacity
        ]

        optim = torch.optim.Adam(params, 1e-2)
        
        camera_params = {
            "params": [self.camera_transform.rotation_quat, self.camera_transform.translate_vec],
            "lr": 3e-5
        }
        optim.add_param_group(camera_params)

        def _closure():
            optim.zero_grad()
            loss = self._loss(images)
            loss.backward()
            return loss

        for i in range(num_steps):
            optim.step(_closure)
            if i % 10 == 0:
                opacity = self.opacity.detach().cpu().numpy()
                occupied = np.nonzero(opacity > 0.5)[0]
                print(i, self._loss(images).item(), len(occupied))
                self.log.append((
                    i,
                    self.camera_transform.rotation_quat.detach().cpu().numpy(),
                    self.camera_transform.translate_vec.detach().cpu().numpy(),
                    occupied))


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


def _carve_and_calibrate(image_dir):
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
        rotation = np.linalg.inv(sp.Transforms.look_at_rotation([0, 0, 0], position, [0, -1, 0])) #np.eye(4, dtype=np.float32)
        camera_to_world = sp.Transforms.translate(position) @ rotation
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = width
        intrinsics[0, 2] = width / 2
        intrinsics[1, 1] = width
        intrinsics[1, 2] = height / 2
        cameras.append(CameraInfo.create("cam{}".format(i), (width, height), intrinsics, camera_to_world))

    occupancy = Occupancy(cameras)
    occupancy.to(device)

    occupancy.fit(images)
    entries = {}
    entries["num_entries"] = len(occupancy.log)
    entries["voxel_depth"] = occupancy.voxels.depth
    entries["iterations"] = np.stack([entry[0] for entry in occupancy.log])
    entries["intrinsics"] = occupancy.camera_transform.intrinsics
    entries["rotation_quats"] = np.stack([entry[1] for entry in occupancy.log])
    entries["translate_vecs"] = np.stack([entry[2] for entry in occupancy.log])
    for i, _, _, occ in occupancy.log:
        entries["occupancy{}".format(i)] = occ

    np.savez("carving.npz", **entries)


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
    canvas = scene.create_canvas_3d()
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
        mesh.add_camera_image(camera)
        image_meshes.append(mesh)

        frustums.add_camera_frustum(camera, colors[i])

    voxel_mesh = scene.create_mesh("voxels", shared_color=sp.Colors.White)
    voxels = OctTree(6)
    positions = voxels.leaf_centers()
    scale = pow(2, -4)
    for index in data["occupancy990"]:
        voxel_mesh.add_cube(transform=sp.Transforms.translate(positions[index]) @ sp.Transforms.scale(scale))

    for i in range(num_cameras):
        frame = canvas.create_frame(camera=cameras[i])
        frame.add_mesh(frustums)
        frame.add_mesh(voxel_mesh)
        for mesh in image_meshes:
            frame.add_mesh(mesh)

    frame.add_mesh(frustums)
    scene.save_as_html("carve.html")   


if __name__ == "__main__":
    image_dir = "D:\\Data\\dinosaur"
    _create_carve_scenepic(image_dir)
    # IT WORKS
    # next steps:
    # 1. at regular intervals do the following:
    #   a. mark voxels which have 0 opacity and prune
    #   b. find voxels which have > 0.5 opacity and expand
    