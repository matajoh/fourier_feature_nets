"""Module containing camera logic."""

from typing import NamedTuple

import numpy as np
import scenepic as sp


def normalize(x):
    """Normalizes a tensor per row."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


Ray = NamedTuple("Ray", [("origin", np.ndarray),
                         ("direction", np.ndarray)])


class Resolution(NamedTuple("Resolution", [("width", int), ("height", int)])):
    """Class representing the width and height of an image."""
    def scale_to_height(self, height: int) -> "Resolution":
        """Scales this resolution while maintaining the aspect ratio.

        Args:
            height (int): The desired new height

        Returns:
            a resolution with the specified height but the same aspect ratio
        """
        width = self.width * height // self.height
        return Resolution(width, height)

    def square(self) -> "Resolution":
        """Returns a square version of this resolution."""
        size = min(self.width, self.height)
        return Resolution(size, size)

    @property
    def ratio(self) -> float:
        """Aspect ratio."""
        return self.width / self.height


class CameraInfo(NamedTuple("CameraInfo", [("name", str),
                                           ("resolution", Resolution),
                                           ("intrinsics", np.ndarray),
                                           ("extrinsics", np.ndarray)])):
    """Encapsulates calibration information about a camera and performs image rectification."""

    @staticmethod
    def create(name: str, resolution: Resolution, intrinsics: np.ndarray,
               extrinsics: np.ndarray) -> "CameraInfo":
        """Creates a default CameraInfo object.

        Arguments:
            name (str): name for the camera
            resolution (Resolution): the resolution of the camera images
            intrinsics (np.ndarray): the 3x3 intrinsics (projection matrix)
            extrinsics (np.ndarray): the 4x4 extrinsics (camera to world matrix)

        Returns:
            CameraInfo: a new camera object
        """
        intrinsics = intrinsics[:3, :3]
        return CameraInfo(name, resolution, intrinsics, extrinsics)

    def unproject(self, points: np.ndarray) -> np.ndarray:
        """Unprojects a series of 2D points to their 3D positions."""
        projection = np.eye(4, dtype=np.float32)
        projection[:3, :3] = self.intrinsics
        projection = projection @ np.linalg.inv(self.extrinsics)
        unprojection = np.linalg.inv(projection)
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
        return points

    @property
    def fov_y_degrees(self) -> float:
        """Y-axis field of view (in degrees) for the camera."""
        fov_y = (0.5 * self.resolution.width) / self.intrinsics[1, 1]
        fov_y = 2 * np.arctan(fov_y)
        return fov_y * 180 / np.pi

    @property
    def position(self) -> np.ndarray:
        """Returns the position of the camera in world coordinates."""
        return self.extrinsics[:3, 3].reshape(1, 3)

    def raycast(self, points: np.ndarray) -> Ray:
        """Casts rays into the world starting corresponding to the specific 2D point positions.

        Arguments:
            points: an array of 2D points in the image plane
        """
        points = points.astype(np.float32)
        world_coords = self.unproject(points)
        camera_pos = self.position
        ray_dir = normalize(world_coords[:, :3] - camera_pos)
        return Ray(camera_pos + 0 * ray_dir, ray_dir)

    def to_scenepic(self, znear=0.01, zfar=100) -> sp.Camera:
        """Creates a ScenePic camera from this camera."""
        world_to_camera = sp.Transforms.gl_world_to_camera(self.extrinsics)
        projection = sp.Transforms.gl_projection(self.intrinsics,
                                                 self.resolution.width,
                                                 self.resolution.height,
                                                 znear, zfar)
        return sp.Camera(world_to_camera, projection)
