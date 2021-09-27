"""Script for preparing Intrinsic3d dataset for use with NeRF"""

import argparse
from collections import namedtuple
import os

import cv2
import numpy as np
from nerf2d import CameraInfo, Triangulation
import svt


def _load_matrix(path: str) -> np.ndarray:
    with open(path) as file:
        rows = []
        for line in file:
            row = [float(val) for val in line.split()]
            rows.append(row)
    
    return np.array(rows, np.float32)


def _extract_keypoints():
    dataset_dir = "D:\\Data\\intrinsic3d\\tomb-statuary-rgbd"
    frame = 0
    sift = cv2.SIFT_create()
    data = {}
    data["intrinsics"] = _load_matrix(os.path.join(dataset_dir, "colorIntrinsics.txt"))
    while True:
        image_path = os.path.join(dataset_dir, "frame-{:06}.color.png".format(frame))
        pose_path = os.path.join(dataset_dir, "frame-{:06}.pose.txt".format(frame))

        if not os.path.exists(image_path):
            break

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)
        points = np.array([kp.pt for kp in keypoints], np.int32)
        data["keypoints{:04}".format(frame)] = points
        colors = image[points[:, 1], points[:, 0]]
        data["colors{:04}".format(frame)] = colors
        data["descriptors{:04}".format(frame)] = descriptors
        data["pose{:04}".format(frame)] = _load_matrix(pose_path)
        img2 = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)
        cv2.imshow("keypoints", img2)
        cv2.waitKey(1)

        frame += 1

    data["resolution"] = (width, height)
    data["num_frames"] = frame
    np.savez("D:\\Data\\intrinsic3d\\tomb_statuary_keypoints.npz", **data)


def _draw_matches(match_image, image0, image1, keypoints0, keypoints1, mask):
    width = image0.shape[1]
    match_image[:, :width] = image0
    match_image[:, width:] = image1
    mask = mask.reshape(-1)
    keypoints0 = keypoints0[mask == 1].astype(np.int32).reshape(-1, 2)
    keypoints1 = keypoints1[mask == 1].astype(np.int32).reshape(-1, 2)
    keypoints1 = keypoints1 + (width, 0)
    for kp0, kp1 in zip(keypoints0, keypoints1):
        pt0 = tuple(kp0)
        pt1 = tuple(kp1)
        cv2.circle(match_image, pt0, 3, (0, 0, 255), -1)
        cv2.circle(match_image, pt1, 3, (0, 0, 255), -1)
        cv2.line(match_image, pt0, pt1, (0, 255, 0))


PathPoint = namedtuple("PathPoint", ["frame", "position", "color"])


def _read_image_scaled(path):
    image = cv2.imread(path)
    image = cv2.GaussianBlur(image, (0, 0), 4)
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)
    return image


def _build_dataset():
    bf = cv2.BFMatcher()

    dataset_dir = "D:\\Data\\intrinsic3d\\tomb-statuary-rgbd"
    data = np.load("D:\\Data\\intrinsic3d\\tomb_statuary_keypoints.npz")
    intrinsics = data["intrinsics"]
    resolution = data["resolution"]
    points = []
    colors = []
    scale = 320 / resolution[0]
    resolution = (320, 240)
    intrinsics[:2] *= scale
    image0 = _read_image_scaled(os.path.join(dataset_dir, "frame-{:06}.color.png".format(0)))
    keypoints0 = data["keypoints{:04}".format(0)] * scale
    descriptors0 = data["descriptors{:04}".format(0)]
    extrinsics0 = data["pose{:04}".format(0)]
    colors0 = data["colors{:04}".format(0)]
    camera = CameraInfo.create("cam0", resolution, intrinsics, extrinsics0)
    match_image = np.zeros((image0.shape[0], image0.shape[1] * 2, 3), np.uint8)
    label_start = 0
    label_end = keypoints0.shape[0]
    labels = np.arange(label_start, label_end)
    paths = {}
    cameras = [camera]
    frames = [image0]
    extrinsics = [extrinsics0]
    for frame in range(1, data["num_frames"]):
        print(frame)
        image1 = _read_image_scaled(os.path.join(dataset_dir, "frame-{:06}.color.png".format(frame)))
        keypoints1 = data["keypoints{:04}".format(frame)] * scale
        descriptors1 = data["descriptors{:04}".format(frame)]
        extrinsics1 = data["pose{:04}".format(frame)]
        colors1 = data["colors{:04}".format(frame)]
        cameras.append(CameraInfo.create("cam1", resolution, intrinsics, extrinsics1))
        extrinsics.append(extrinsics0)
        frames.append(image1)

        matches = bf.knnMatch(descriptors0, descriptors1, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)    

        src_pts = np.array([keypoints0[m.queryIdx] for m in good], np.float32).reshape(-1,1,2)
        dst_pts = np.array([keypoints1[m.trainIdx] for m in good], np.float32).reshape(-1,1,2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        mask = mask.reshape(-1)
        label_start = label_end
        label_end = label_start + keypoints1.shape[0]
        new_labels = np.arange(label_start, label_end)
        for m, valid in zip(good, mask):
            if not valid:
                continue

            q_label = labels[m.queryIdx]
            if not q_label in paths:
                paths[q_label] = [PathPoint(frame - 1, keypoints0[m.queryIdx], colors0[m.queryIdx])]

            paths[q_label].append(PathPoint(frame, keypoints1[m.trainIdx], colors1[m.trainIdx]))
            new_labels[m.trainIdx] = q_label

        labels = new_labels

        _draw_matches(match_image, image0, image1, src_pts, dst_pts, mask)
        cv2.imshow("matches", match_image)
        cv2.waitKey(1)

        image0 = image1
        keypoints0 = keypoints1
        descriptors0 = descriptors1
        extrinsics0 = extrinsics1
        colors0 = colors1

    cv2.destroyAllWindows()

    print("Triangulating points")
    size = np.array(resolution).astype(np.float32)
    points = []
    colors = []
    min_path_length = 40
    for ldmk_id in paths:
        path = paths[ldmk_id]
        if len(path) < min_path_length:
            continue

        print(len(points))

        landmarks = np.stack([p.position for p in path])
        landmarks = (2 * (landmarks + 0.5)) / size - 1
        camera_info = [cameras[p.frame] for p in path]
        landmarks = landmarks.reshape(1, -1, 1, 2)
        triangulation = Triangulation(camera_info)
        point = triangulation(landmarks).reshape(3)

        points.append(point)

        path_colors = np.stack([p.color for p in path])
        colors.append(path_colors.mean(0).astype(np.uint8))

    print("done")
    np.savez("D:\\Data\\intrinsic3d\\tomb_statuary.npz",
             points=points,
             colors=colors,
             frames=frames,
             intrinsics=intrinsics,
             extrinsics=extrinsics,
             resolution=resolution)


def _create_svt():
    data = np.load("D:\\Data\\intrinsic3d\\tomb_statuary.npz")
    points = data["points"]
    colors = data["colors"].astype(np.float32) / 255
    frames = data["frames"]
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]
    resolution = data["resolution"]

    focus_point = points.mean(0)

    scene = svt.Scene()
    canvas = scene.create_canvas_3d(width=800, height=600)
    mesh = scene.create_mesh()
    mesh.add_icosphere(color=svt.Colors.Magenta, transform=svt.Transforms.scale(0.01))
    mesh.enable_instancing(points, colors=colors[:, ::-1])

    num_frames = frames.shape[0]
    for i in range(num_frames):
        print(i, "/", num_frames)
        image = scene.create_image()
        image.from_numpy(frames[i, :, :, ::-1])

        camera = CameraInfo.create("cam", resolution, intrinsics, extrinsics[i])
        svt_camera = camera.to_svt()

        frustum_mesh = scene.create_mesh(layer_id="frustums")
        frustum_mesh.add_camera_frustum(svt_camera, svt.Colors.White, thickness=0.005, depth=0.2)

        image_mesh = scene.create_mesh(shared_color=svt.Colors.White, double_sided=True, layer_id="images")
        image_mesh.texture_id = image.image_id
        image_mesh.add_camera_image(svt_camera, depth=0.2)

        frame = canvas.create_frame(camera=svt_camera, focus_point=focus_point)
        frame.add_mesh(mesh)
        frame.add_mesh(frustum_mesh)
        frame.add_mesh(image_mesh)

    scene.save_as_html("D:\\Data\\intrinsic3d\\tomb_statuary.html")


def _main():
    #_extract_keypoints()
    _build_dataset()
    _create_svt()

   

if __name__ == "__main__":
    _main()
