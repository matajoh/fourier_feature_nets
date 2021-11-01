"""Utility module."""

from progress.bar import Bar
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
