"""Utility module."""

import base64
import os
from typing import List

import numpy as np
from progress.bar import Bar
import requests
import scenepic as sp
import torch

from .camera_info import CameraInfo, Resolution
from .fourier_feature_models import FourierFeatureMLP
from .nerf_model import NeRF
from .voxels_model import Voxels


def _in_ipynb():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    except ModuleNotFoundError:
        return False


class ETABar(Bar):
    """Progress bar that displays the estimated time of completion."""
    suffix = "%(percent).1f%% - %(eta)ds"
    bar_prefix = " "
    bar_suffix = " "
    empty_fill = "∙"
    fill = "█"

    def writeln(self, line: str):
        """Writes the line to the console.

        Description:
            This method is Jupyter notebook aware, and will do the
            right thing when in that environment as opposed to being
            run from the command line.

        Args:
            line (str): The message to write
        """
        if _in_ipynb():
            from IPython.display import clear_output
            clear_output(wait=True)
            print(line)
        else:
            Bar.writeln(self, line)

    def info(self, text: str):
        """Appends the given information to the progress bar message.

        Args:
            text (str): A status message for the progress bar.
        """
        self.suffix = "%(percent).1f%% - %(eta)ds {}".format(text)


def calculate_blend_weights(t_values: torch.Tensor,
                            opacity: torch.Tensor) -> torch.Tensor:
    """Calculates blend weights for a ray.

    Args:
        t_values (torch.Tensor): A (num_rays,num_samples) tensor of t values
        opacity (torch.Tensor): A (num_rays,num_samples) tensor of opacity
                                opacity values for the ray positions.

    Returns:
        torch.Tensor: A (num_rays,num_samples) blend weights tensor
    """
    _, num_samples = t_values.shape
    deltas = t_values[:, 1:] - t_values[:, :-1]
    max_dist = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, max_dist], dim=-1)

    alpha = 1 - torch.exp(-(opacity * deltas))
    ones = torch.ones_like(alpha)

    trans = torch.minimum(ones, 1 - alpha + 1e-10)
    trans, _ = trans.split([num_samples - 1, 1], dim=-1)
    trans = torch.cat([torch.ones_like(trans[:, :1]), trans], dim=-1)
    trans = torch.cumprod(trans, -1)
    weights = alpha * trans
    return weights


ASSETS = {
    "antinous_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluBagOAnmTej7LJb_Q",
    "antinous_800.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluIjnhVcVei5mZMIpw",
    "benin_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluIX9MtESyi1LX9L8Q",
    "benin_800.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluIlZRDTjHdSQnt_2A",
    "lego_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluBbbdxzOG5q4a98yA",
    "lego_800.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluIb8oRozVWUMQCfmg",
    "matthew_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluIz9A0gFTi-yBs8zQ",
    "matthew_800.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluI0rBTyq9jSnd4IjA",
    "rubik_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluI60mrfqAcxYIsdLg",
    "rubik_800.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluI7DdBRXbBngRMEew",
    "trex_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluM59kAfIq0H1AVdQA",
    "trex_800.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluM63nCZzfryxRR7ow",
    "antinous_800_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJLoo7yjPYQz8W5dg",
    "antinous_800_nerf.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJclttRvj65vHpUiA",
    "benin_800_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJNUlKmPZJiZ3HUlg",
    "benin_800_nerf.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJd2newCq4oVIlrXw",
    "lego_800_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJM8A6nLNsSxgaZLw",
    "lego_800_nerf.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJeY79jz1o51K4CIg",
    "matthew_800_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJOcOc6Ce3ZUcQl3g",
    "matthew_800_nerf.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJf0KKODbTR291vwQ",
    "trex_800_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluM74RKvya3PjvzqTw",
    "antinous_400_mlp.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJqpdzBhx9QAtbJ-g",
    "antinous_400_pos.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJokd4Fl4UGLI_bNw",
    "benin_400_mlp.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJlhzc0JjMUus5HsA",
    "benin_400_pos.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJkAd3De0s2DR_RoA",
    "lego_400_mlp.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJnRCQdmHfJiXvGNw",
    "lego_400_pos.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJrPmpRYZlP0fP5Eg",
    "matthew_400_mlp.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJmKYDqQpitLHVIHg",
    "matthew_400_pos.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluJpn1o7zC8uhdSDXA",
    "antinous_octree_8.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluJt3FR8NAJW84HT2A",
    "antinous_octree_10.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluJupuBKuwq0hYk-Tw"
}


def _create_onedrive_directdownload(onedrive_link: str):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, "utf-8"))
    data_bytes64 = data_bytes64.decode("utf-8")
    data_bytes64 = data_bytes64.replace("/", "_").replace("+", "-").rstrip("=")
    return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64}/root/content"


def download_asset(name: str, output_path: str) -> bool:
    """Downloads one of the known datasets.

    Args:
        name (str): Key for the ASSETS dictionary
        output_path (str): Path to the downloaded NPZ

    Returns:
        bool: whether the download was successful
    """
    if name not in ASSETS:
        print("Unrecognized asset:", name)
        return False

    print("Downloading", name, "to", output_path)
    url = _create_onedrive_directdownload(ASSETS[name])
    res = requests.get(url, stream=True)
    total_bytes = int(res.headers.get("content-length"))
    bar = ETABar("Downloading", max=total_bytes)
    report_interval = 32
    chunks = 0
    with open(output_path, "wb") as file:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                if chunks % report_interval == 0:
                    bar.next(len(chunk) * report_interval)

                chunks += 1
                file.write(chunk)

    bar.finish()
    return True


def linspace(start: torch.Tensor, stop: torch.Tensor,
             num_samples: int) -> torch.Tensor:
    """Generalization of linspace to arbitrary tensors.

    Args:
        start (torch.Tensor): Minimum 1D tensor. Same length as stop.
        stop (torch.Tensor): Maximum 1D tensor. Same length as start.
        num_samples (int): The number of samples to take from the linear range.

    Returns:
        torch.Tensor: (D, num_samples) tensor of linearly interpolated
                      samples.
    """
    diff = stop - start
    samples = torch.linspace(0, 1, num_samples)
    return start.unsqueeze(-1) + samples.unsqueeze(0) * diff.unsqueeze(-1)


def interpolate_bilinear(grid: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """Find the values of a function using bilinear interpolation.

    Given a vector-valued function of two variables defined on a rectangular grid, find
    the values of that function for each point in query_points.

    Argumentss:
        grid: Tensor of shape (height, width, dim) containing the values of the function
              on a grid.
        query_points: Tensor of shape (N, 2) containing N query points
                      normalized between 0 and 1.

    Returns:
        A tensor of shape (N, dim) containing the value of the function for each query point.
    """
    assert len(grid.shape) == 3, "Grid has to be of shape (height, width, dim)"
    assert len(query_points.shape) == 2, "Query points have to be of shape (N, 2)"

    height, width, _ = grid.shape

    col = query_points[:, 0] * width
    row = query_points[:, 1] * height
    i0 = np.floor(row).astype(np.int32)
    j0 = np.floor(col).astype(np.int32)
    di = row - i0.astype(row.dtype)
    dj = col - j0.astype(col.dtype)
    i1 = i0 + 1
    j1 = j0 + 1

    i0 = np.clip(i0, 0, height - 1)
    j0 = np.clip(j0, 0, width - 1)
    i1 = np.clip(i1, 0, height - 1)
    j1 = np.clip(j1, 0, width - 1)

    di = np.expand_dims(di, -1)
    dj = np.expand_dims(dj, -1)

    v00 = (1-di)*(1-dj)*grid[i0, j0, :]
    v01 = (1-di)*dj*grid[i0, j1, :]
    v10 = di*(1-dj)*grid[i1, j0, :]
    v11 = di*dj*grid[i1, j1, :]
    result = v00 + v01 + v10 + v11
    result = result.reshape(-1, grid.shape[-1])

    return result


def orbit(up_dir: np.ndarray, forward_dir: np.ndarray, num_frames: int,
          fov_y_degrees: float, resolution: Resolution,
          distance: float, min_altitude=np.pi/12,
          max_altitude=np.pi/4) -> List[CameraInfo]:
    """Computes a list of cameras forming an orbit around the origin.

    Args:
        up_dir (np.ndarray): The up direction that forms the main axis of
                             rotation.
        forward_dir (np.ndarray): Used to place the initial camera location
                                  and determine the secondary axis of rotation.
        num_frames (int): The number of positions in the orbit.
        fov_y_degrees (float): Field of view for the camera.
        resolution (Resolution): Resolution of the camera.
        distance (float): Distance of the camera from the origin.
        min_altitude (float, optional): Minimum camera altitude in radians.
                                        Defaults to pi/12.
        max_altitude (float, optional): Maximum camera altitude in radians.
                                        Defaults to pi/4.

    Returns:
        List[CameraInfo]: The computed camera positions for the orbit.
    """
    right_dir = np.cross(up_dir, forward_dir)

    azimuth = np.linspace(0, 4 * np.pi, num_frames, endpoint=False)

    altitude = np.zeros_like(azimuth)
    half_frames = num_frames // 2
    altitude[:half_frames] = np.linspace(min_altitude, max_altitude,
                                         half_frames, endpoint=False)
    altitude[half_frames:] = np.linspace(max_altitude, min_altitude,
                                         num_frames - half_frames,
                                         endpoint=False)

    fov_y = fov_y_degrees * np.pi / 180
    focal_length = .5 * resolution.width / np.tan(.5 * fov_y)

    intrinsics = np.array([
        focal_length, 0, resolution.width / 2,
        0, focal_length, resolution.height / 2,
        0, 0, 1
    ], np.float32).reshape(3, 3)

    camera_info = []
    camera = sp.Camera(-forward_dir * distance, up_dir=up_dir)
    init_ext = camera.camera_to_world @ sp.Transforms.rotation_about_x(np.pi)
    for frame_azi, frame_alt in zip(azimuth, altitude):
        elevate = sp.Transforms.rotation_matrix_from_axis_angle(right_dir,
                                                                frame_alt)
        rotate = sp.Transforms.rotation_matrix_from_axis_angle(up_dir,
                                                               frame_azi)

        extrinsics = rotate @ elevate @ init_ext
        camera = CameraInfo.create("cam{}".format(len(camera_info)),
                                   resolution,
                                   intrinsics, extrinsics)
        camera_info.append(camera)

    return camera_info


def exponential_lr_decay(optim: torch.optim.Adam,
                         initial_learning_rate: float,
                         step: int, decay_rate: float,
                         decay_steps: float):
    """Keras-style per-step learning rate decay.

    Description:
        This method will decay the learning rate continuously for all
        parameter groups in the optimizer by the function:

        lr = lr_0 * decay^(step / decay_steps)

    Args:
        optim (torch.optim.Adam): The optimizer to modify
        initial_learning_rate (float): The initial learning rate (lr_0)
        step (int): The current step in training
        decay_rate (float): The rate at which to decay the learning rate
        decay_steps (float): The number of steps before the learning rate is
                             applied in full.
    """
    decay_rate = decay_rate ** (step / decay_steps)
    lr = initial_learning_rate * decay_rate
    for group in optim.param_groups:
        group["lr"] = lr


def load_model(path: str) -> torch.nn.Module:
    """Loads a supported model from the path.

    Description:
        This method will attempt to load a model from the path provided.
        If it cannot find it, it will look around in common locations, and then
        attempt to download it (if it is a known asset) to the provided
        path. If successful, it will then determine what kind of model it is
        and return the correct model object in evaluation mode.

    Args:
        path (str): Path to the model file

    Returns:
        torch.nn.Module: An initialized model file in evaluation mode.
    """
    if not os.path.exists(path):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        path = os.path.join(models_dir, path)
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print("Downloading model...")
            model_name = os.path.basename(path)
            success = download_asset(model_name, path)
            if not success:
                print("Unable to download model", model_name)
                return None

    state_dict = torch.load(path)
    if state_dict["type"] == "fourier":
        model_class = FourierFeatureMLP
        params = state_dict["params"]
        if params["a_values"] is not None:
            a_values = params["a_values"]
            a_values = torch.FloatTensor(a_values)
            params["a_values"] = a_values
        if params["b_values"] is not None:
            b_values = params["b_values"]
            b_values = torch.FloatTensor(b_values)
            params["b_values"] = b_values
    elif state_dict["type"] == "nerf":
        model_class = NeRF
    elif state_dict["type"] == "voxels":
        model_class = Voxels
    else:
        print("Unrecognized model type:", state_dict["type"])

    del state_dict["type"]
    model = model_class(**state_dict["params"])
    del state_dict["params"]
    model.load_state_dict(state_dict)
    model.eval()
    return model
