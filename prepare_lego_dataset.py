import argparse
import json
import math
import os
from typing import List

import cv2
from nerf import CameraInfo
import numpy as np
import scenepic as sp


def _convert_cameras(image_dir, split):
    print("Converting", split, "cameras")
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
            aspect_ratio = width / height
            focal_length = .5 * width / math.tan(.5 * fov_x)
            intrinsics = np.eye(3, dtype=np.float32)
            intrinsics[0, 0] = focal_length
            intrinsics[0, 2] = width / 2
            intrinsics[1, 1] = focal_length / aspect_ratio
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
        label = scene.create_label(text=camera.name, size_in_pixels=40)
        camera_labels.append(label)

        image = scene.create_image()
        image.from_numpy(pixels)
        image_mesh = scene.create_mesh(layer_id="images", texture_id=image.image_id, double_sided=True)
        image_mesh.add_camera_image(sp_camera, depth=0.5)
        image_meshes.append(image_mesh)

    for camera in cameras:
        sp_camera = camera.to_scenepic()

        frame = canvas.create_frame()
        frame.camera = sp_camera
        frame.add_mesh(cube_mesh)
        frame.add_mesh(frustums)
        for camera, label, image_mesh in zip(cameras, camera_labels, image_meshes):
            frame.add_label(label, camera.position)
            frame.add_mesh(image_mesh)

    scene.save_as_html(os.path.join(image_dir, "{}_cameras.html".format(split)))
    CameraInfo.to_json(os.path.join(image_dir, "{}_cameras.json".format(split)), cameras)


def _extract_masks(image_dir):
    masks_dir = os.path.join(image_dir, "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    for name in os.listdir(image_dir):
        if not name.endswith(".png"):
            continue

        mask_path = os.path.join(masks_dir, name)
        image = cv2.imread(os.path.join(image_dir, name))

        chroma = [0, 0, 0]
        test = (image == chroma).sum(-1)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[test != 3] = 255
        cv2.imwrite(mask_path, mask)


def _parse_args():
    parser = argparse.ArgumentParser("Prepare the lego dataset")
    parser.add_argument("data_dir")
    return parser.parse_args()


def _main():
    # TODO add download and unzip
    args = _parse_args()
    for split in ["train", "val", "test"]:
        _convert_cameras(args.data_dir, split)

    _extract_masks(os.path.join(args.data_dir, "train"))


if __name__ == "__main__":
    _main()
