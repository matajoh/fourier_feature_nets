"""Module containing camera logic."""

from collections import namedtuple
import json
from typing import List, Tuple

import numpy as np
import scenepic as sp


def normalize(x):
    """Normalizes a tensor per row."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


Ray = namedtuple("Ray", ["origin", "direction"])


class CameraInfo(namedtuple("CameraInfo", ["name", "resolution", "intrinsics", "extrinsics"])):
    """Encapsulates calibration information about a camera and performs image rectification."""

    def update(self, intrinsics: np.ndarray, extrinsics: np.ndarray) -> "CameraInfo":
        """Updates the extrinsic and intrinsic settings for the camera and returns a new CameraInfo object."""
        return CameraInfo(self.name, self.resolution, intrinsics, extrinsics)

    def scale(self, scale: float) -> "CameraInfo":
        width, height = self.resolution
        intrinsics = self.intrinsics * scale
        resolution = width * scale, height * scale
        return CameraInfo(self.name, resolution, intrinsics, self.extrinsics)

    @staticmethod
    def from_json(path: str) -> List["CameraInfo"]:
        """Constructs a list of CameraInfo objects from a JSON file (produces by `to_json`)."""
        with open(path) as file:
            cameras = json.load(file)

        camera_info = []
        for camera in cameras:
            resolution = tuple(camera["resolution"])
            intrinsics = np.stack(camera["intrinsics"]).astype(np.float32)
            extrinsics = np.stack(camera["extrinsics"]).astype(np.float32)

            camera_info.append(CameraInfo(camera["name"], resolution, intrinsics, extrinsics))

        return camera_info

    @staticmethod
    def to_json(path: str, camera_info: List["CameraInfo"]):
        """Writes a list of CameraInfo objects in JSON format to a file."""
        camera_dicts = []
        for camera in camera_info:
            state_dict = {}
            state_dict["name"] = camera.name
            state_dict["resolution"] = tuple(camera.resolution)
            state_dict["intrinsics"] = camera.intrinsics.tolist()
            state_dict["extrinsics"] = camera.extrinsics.tolist()
            camera_dicts.append(state_dict)

        with open(path, "w") as file:
            json.dump(camera_dicts, file)

    @staticmethod
    def create(name: str, resolution: Tuple[int, int], intrinsics: np.ndarray, extrinsics: np.ndarray) -> "CameraInfo":
        """Creates a default CameraInfo object using the `data` argument as a reference.

        Arguments:
            name (str): name for the camera
            resolution (Tuple[int, int]): the resolution of the camera images
            intrinsics (np.ndarray): the 3x3 intrinsics
            extrinsics (np.ndarray): the 4x4 extrinsics (camera to world matrix)

        Returns:
            CameraInfo: a new camera object
        """
        resolution = tuple(resolution)
        intrinsics = intrinsics[:3, :3]
        return CameraInfo(name, resolution, intrinsics, extrinsics)

    def unproject(self, points: np.ndarray) -> np.ndarray:
        """Unprojects a series of 2D points to their 3D positions."""
        projection = np.eye(4, dtype=np.float32)
        projection[:3, :3] = self.intrinsics
        projection = projection @ np.linalg.inv(self.extrinsics)
        unprojection = np.linalg.inv(projection)
        center = (np.array(self.resolution, np.float32) - 1) / 2
        center = center.reshape(1, 1, 2)
        points = (points * center) + center
        h_coords = points.reshape(-1, 2)
        h_coords = np.concatenate([h_coords, np.ones((h_coords.shape[0], 2), np.float32)], axis=-1)
        return (unprojection @ h_coords.T).T

    def project(self, positions: np.ndarray) -> np.ndarray:
        """Projects 3D positions into 2D points in the image plane."""
        projection = np.eye(4, dtype=np.float32)
        projection[:3, :3] = self.intrinsics
        projection = projection @ np.linalg.inv(self.extrinsics)
        ones = np.ones((positions.shape[0], 1), np.float32)
        h_coords = np.concatenate([positions, ones], -1)
        points = (projection @ h_coords.T).T
        points = points[:, :2] / points[:, 2:3]
        center = (np.array(self.resolution, np.float32) - 1) / 2
        center = center.reshape(1, 1, 2)
        points = (points - center) / center
        return points

    @property
    def position(self) -> np.ndarray:
        return self.extrinsics[:3, 3].reshape(1, 3)

    def raycast(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Casts rays into the world starting corresponding to the specific 2D point positions.

        Arguments:
            points: an array of 2D points in the image plane
            znear: the z distance of the ray starting point (default: 1)
            zfar: the z distance of the ray ending point (default: 100)
        """
        world_coords = self.unproject(points)
        camera_pos = self.position
        ray_dir = normalize(world_coords[:, :3] - camera_pos)
        return camera_pos + 0 * ray_dir, ray_dir

    def to_scenepic(self, znear=0.01, zfar=100) -> sp.Camera:
        """Creates a ScenePic camera from this camera."""
        world_to_camera = sp.Transforms.gl_world_to_camera(self.extrinsics)
        projection = sp.Transforms.gl_projection(self.intrinsics, self.resolution[0], self.resolution[1], znear, zfar)
        return sp.Camera(world_to_camera, projection)
