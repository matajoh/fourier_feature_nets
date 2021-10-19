import argparse
import os
import sys

import cv2
from matplotlib.pyplot import get_cmap
from nerf import CameraInfo, OcTree, RaySamplingDataset
import numpy as np
import scenepic as sp


def _parse_args():
    parser = argparse.ArgumentParser("Ray Sampling Tester")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--voxels-dir", help="Path to the voxels directory")
    parser.add_argument("--path-length", type=int, default=128,
                        help="Number of voxels to intersect")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=10,
                        help="Number of cameras")
    return parser.parse_args()


def _main():
    args = _parse_args()
    image_dir = os.path.join(args.data_dir, "train")
    images = []
    cameras = CameraInfo.from_json(os.path.join(args.data_dir, "train_cameras.json"))
    cameras = cameras[:args.num_cameras]
    for camera in cameras:
        image = cv2.imread(os.path.join(image_dir, camera.name + ".png"))
        images.append(image[:, :, ::-1])

    images = np.stack(images)
    _, width, height, _ = images.shape
    path_length = args.path_length
    if args.voxels_dir:
        data = np.load(os.path.join(args.voxels_dir, "carving.npz"))
        voxels = OcTree.load(data)
        opacity = data["opacity"]
    else:
        voxels = None
        opacity = None

    dataset = RaySamplingDataset(images, cameras, args.num_samples, 
                                 args.resolution, path_length, voxels, opacity,
                                 stratified=True)

    scene = sp.Scene()
    frustums = scene.create_mesh("frustums", layer_id="frustums")
    canvas = scene.create_canvas_3d(width=width, height=height)
    cmap = get_cmap("jet")
    camera_colors = cmap(np.linspace(0, 1, args.num_cameras))[:, :3]
    image_meshes = []
    sys.stdout.write("Adding cameras")
    for pixels, camera, color in zip(images, cameras, camera_colors):
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
    num_rays = args.resolution * args.resolution
    for i, camera in enumerate(cameras):
        sys.stdout.write(".")
        sys.stdout.flush()
        start = i * num_rays
        end = start + num_rays
        index = [i for i in range(start, end)]
        positions, _, _, colors = dataset[index]

        colors = colors.unsqueeze(1).expand(-1, args.num_samples, -1)
        positions = positions.numpy().reshape(-1, 3)
        colors = colors.numpy().copy().reshape(-1, 3)

        empty = (colors == 0).sum(-1) == 3
        not_empty = np.logical_not(empty)

        samples = scene.create_mesh()
        samples.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.02))
        samples.enable_instancing(positions=positions[not_empty],
                                  colors=colors[not_empty])

        empty_samples = scene.create_mesh(layer_id="empty",
                                          shared_color=sp.Colors.Gray)
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
    scene.save_as_html("ray_sampling.html", "Ray Sampling")


if __name__ == "__main__":
    _main()
