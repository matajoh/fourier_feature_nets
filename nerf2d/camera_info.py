from collections import namedtuple
import json
from typing import List, Tuple

import cv2
import numpy as np
import svt


def normalize(x):
    """Normalizes a tensor per row."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class CameraInfo(namedtuple("CameraInfo", ["name", "resolution", "camera_matrix",
                                           "dist_coeffs", "map1", "map2", "intrinsic", "extrinsic"])):
    """Encapsulates calibration information about a camera and performs image rectification."""

    def rectify(self, image: np.ndarray) -> np.ndarray:
        """Rectifies the provided image and returns the result."""
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_CUBIC)

    def update(self, extrinsic: np.ndarray, intrinsic: np.ndarray) -> "CameraInfo":
        """Updates the extrinsic and intrinsic settings for the camera and returns a new CameraInfo object."""
        return CameraInfo(self.name, self.resolution, self.camera_matrix,
                          self.dist_coeffs, self.map1, self.map2, intrinsic, extrinsic)

    @staticmethod
    def from_json(path: str) -> List["CameraInfo"]:
        """Constructs a list of CameraInfo objects from a JSON file (produces by `to_json`)."""
        with open(path) as file:
            cameras = json.load(file)

        camera_info = []
        for camera in cameras:
            camera_matrix = np.array(camera["camera_matrix"], np.float32)
            dist_coeffs = np.array(camera["dist_coeffs"], np.float32)
            resolution = tuple(camera["resolution"])
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                                 dist_coeffs,
                                                                 resolution, 0,
                                                                 centerPrincipalPoint=True)
            map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3),
                                                     new_camera_matrix, resolution, cv2.CV_16SC2)
            if "transform" in camera:
                extrinsic = np.array(camera["transform"], np.float32)
            else:
                extrinsic = np.eye(4, dtype=np.float32)

            camera_info.append(CameraInfo(camera["name"], resolution, camera_matrix, dist_coeffs,
                                          map1, map2, new_camera_matrix, extrinsic))

        return camera_info

    @staticmethod
    def to_json(path: str, camera_info: List["CameraInfo"]):
        """Writes a list of CameraInfo objects in JSON format to a file."""
        camera_dicts = []
        for camera in camera_info:
            state_dict = {}
            state_dict["camera_matrix"] = camera.camera_matrix.tolist()
            state_dict["dist_coeffs"] = camera.dist_coeffs.tolist()
            state_dict["resolution"] = tuple(camera.resolution)
            state_dict["transform"] = camera.extrinsic.tolist()
            state_dict["name"] = camera.name
            camera_dicts.append(state_dict)

        with open(path, "w") as file:
            json.dump(camera_dicts, file)

    @staticmethod
    def create(name: str, resolution: Tuple[int, int], intrinsics: np.ndarray, extrinsics: np.ndarray) -> "CameraInfo":
        """Creates a default CameraInfo object using the `data` argument as a reference.

        Arguments:
            path: path to the output JSON file
            data: Either a (num_cameras, height, width, 3) tensor of images, or a sequence NPZ reference
        """
        resolution = tuple(resolution)
        intrinsics = intrinsics[:3, :3]
        dist_coeffs = np.zeros((5,), np.float32)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(intrinsics,
                                                             dist_coeffs,
                                                             resolution, 0,
                                                             centerPrincipalPoint=True)
        map1, map2 = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, np.eye(3),
                                                 new_camera_matrix, resolution, cv2.CV_16SC2)

        return CameraInfo(name, resolution, intrinsics, dist_coeffs,
                          map1, map2, new_camera_matrix, extrinsics)


    def unproject(self, points: np.ndarray) -> np.ndarray:
        """Unprojects a series of 2D points to their 3D positions."""
        projection = np.eye(4, dtype=np.float32)
        projection[:3, :3] = self.intrinsic
        projection = projection @ np.linalg.inv(self.extrinsic)
        unprojection = np.linalg.inv(projection)
        center = (np.array(self.resolution, np.float32) - 1) / 2
        center = center.reshape(1, 1, 2)
        points = (points * center) + center
        h_coords = points.reshape(-1, 2)
        h_coords = np.concatenate([h_coords, np.ones((h_coords.shape[0], 2), np.float32)], axis=-1)
        return (unprojection @ h_coords.T).T

    def raycast(self, points: np.ndarray, znear=1, zfar=100) -> Tuple[np.ndarray, np.ndarray]:
        """Casts rays into the world starting corresponding to the specific 2D point positions.

        Arguments:
            points: an array of 2D points in the image plane
            znear: the z distance of the ray starting point (default: 1)
            zfar: the z distance of the ray ending point (default: 100)
        """
        world_coords = self.unproject(points)
        camera_pos = self.extrinsic[:3, 3].reshape(1, 3)
        ray_dir = normalize(world_coords[:, :3] - camera_pos)
        points_near = (camera_pos + ray_dir * znear).reshape(points.shape[0], points.shape[1], 3)
        points_far = (camera_pos + ray_dir * zfar).reshape(points.shape[0], points.shape[1], 3)
        return points_near, points_far

    def to_svt(self, znear=0.1, zfar=100) -> svt.Camera:
        """Creates an SVT camera from this camera."""
        world_to_camera = svt.Transforms.gl_world_to_camera(self.extrinsic)
        projection = svt.Transforms.gl_projection(self.intrinsic, self.resolution[0], self.resolution[1], znear, zfar)
        return svt.Camera(world_to_camera, projection)
