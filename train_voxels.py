import argparse
import json
import math
import os
import sys
from typing import List

import cv2
from matplotlib.pyplot import get_cmap
from nerf import CameraInfo, OcTree, VolumeCarver
import numpy as np
import scenepic as sp


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

    voxels = OcTree.load(data)

    scene = sp.Scene()
    frustums = scene.create_mesh("frustums", layer_id="frustums")
    canvas = scene.create_canvas_3d(width=width, height=height)
    cmap = get_cmap("jet")
    image_meshes = []
    cameras = [CameraInfo.create(name, (width, height), intrinsics, extrinsics)
               for name, intrinsics, extrinsics in zip(data["names"],
                                                       data["intrinsics"],
                                                       data["extrinsics"])]
    num_cameras = len(cameras)
    colors = cmap(np.linspace(0, 1, num_cameras))[:, :3]
    sys.stdout.write("Adding cameras")
    for camera, pixels, color in zip(cameras, images, colors):
        sys.stdout.write(".")
        sys.stdout.flush()
        sp_camera = camera.to_scenepic()

        image = scene.create_image(camera.name)
        image.from_numpy(pixels)
        mesh = scene.create_mesh(camera.name, layer_id="images",
                                 texture_id=image.image_id, double_sided=True)
        mesh.add_camera_image(sp_camera, depth=0.5)
        image_meshes.append(mesh)

        frustums.add_camera_frustum(sp_camera, color, depth=0.5, thickness=0.01)

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
        frame = canvas.create_frame(camera=cameras[i].to_scenepic())
        frame.add_mesh(frustums)
        for mesh in voxel_meshes:
            frame.add_mesh(mesh)

        for mesh in image_meshes:
            frame.add_mesh(mesh)

    scene.framerate = 10
    scene.save_as_html(os.path.join(results_dir, "carve.html"), "Voxel Carving")


def _parse_args():
    parser = argparse.ArgumentParser("Volume Dataset Preparation")
    parser.add_argument("image_dir", help="Image containing the training images")
    parser.add_argument("results_dir", help="Output directory for results")
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
    _carve(vars(args))
    _create_carve_scenepic(args.image_dir, args.results_dir)


if __name__ == "__main__":
    _main()