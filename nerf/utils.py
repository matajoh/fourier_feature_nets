"""Utility module."""

import base64

from progress.bar import Bar
import requests
import torch


class ETABar(Bar):
    """Progress bar that displays the estimated time of completion."""
    suffix = "%(percent).1f%% - %(eta)ds"
    bar_prefix = " "
    bar_suffix = " "
    empty_fill = "∙"
    fill = "█"

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
    "lego_400.npz": "https://1drv.ms/u/s!AnWvK2b51nGqluBbbdxzOG5q4a98yA",
    "antinous_400_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluB8NZz3kOP4at1QPQ",
    "lego_400_vox128.pt": "https://1drv.ms/u/s!AnWvK2b51nGqluB7Bf3qx5P3h5V9CQ"
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
    with open(output_path, "wb") as file:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                bar.next(len(chunk))
                file.write(chunk)

    bar.finish()
    return True


def linspace(start: torch.Tensor, stop: torch.Tensor, num_samples: int):
    diff = stop - start
    samples = torch.linspace(0, 1, num_samples)
    return start.unsqueeze(-1) + samples.unsqueeze(0) * diff.unsqueeze(-1)
