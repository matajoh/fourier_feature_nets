"""Module with logic for a 1-D signal dataset."""

from typing import Callable, NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .fourier_feature_models import FourierFeatureMLP


class SignalData(NamedTuple("FunctionData", [("x", torch.FloatTensor),
                                             ("y", torch.FloatTensor)])):
    """1-D Signal data with x and corresponding y values."""


def _get_limits(vals: Union[np.ndarray, torch.Tensor], stretch=1.1):
    min_x, max_x = vals.min().item(), vals.max().item()
    mid_x = 0.5 * (min_x + max_x)
    min_x = mid_x + stretch * (min_x - mid_x)
    max_x = mid_x + stretch * (max_x - mid_x)
    return min_x, max_x


class SignalDataset:
    """Dataset consisting of 1-d signal data."""

    def __init__(self, train_data: SignalData, val_data: SignalData):
        """Constructor.

        Args:
            train_data (SignalData): The x/y values for training
            val_data (SignalData): The x/y values for validation
        """
        self.train_x, self.train_y = train_data
        self.val_x, self.val_y = val_data
        self.x_lim = _get_limits(self.val_x)
        self.y_lim = _get_limits(self.val_y)

    @staticmethod
    def create(signal: Callable[[np.ndarray], np.ndarray],
               num_samples: int, sample_rate: int) -> "SignalDataset":
        """Creates a signal data using the provided signal function.

        Description:
            The signal function should handle values ranging from 0 to 2.

        Args:
            signal (Callable[[np.ndarray], np.ndarray]): This is a 1D signal
                                                         function.
            num_samples (int): The number of samples to use for training.
            sample_rate (int): The rate at which training samples are taken.
                               For example, if a rate of 8 is used then every
                               8th point will be used for training.

        Returns:
            SignalDataset: The constructed dataset
        """
        x = np.linspace(0, 2,
                        num_samples * sample_rate,
                        endpoint=False).astype(np.float32)
        y = signal(x)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        train_data = SignalData(torch.from_numpy(x[::sample_rate]),
                                torch.from_numpy(y[::sample_rate]))
        val_data = SignalData(torch.from_numpy(x), torch.from_numpy(y))
        return SignalDataset(train_data, val_data)

    def plot(self, space_ax: plt.Axes, hidden_ax: plt.Axes,
             model: FourierFeatureMLP, num_points: int,
             colors: np.ndarray, max_hidden: int):
        """Plots a visualization of the model to the given axes.

        Description:
            This method will plot the top N activations from the
            final layer of the MLP (as indicated by `max_hidden`)
            multiplied by the slopes and shifted by the bias
            appropriate to the final regressor, on `hidden_ax`. On
            `space_ax` it will plot the reconstructed sample points
            and the ground truth function.

        Args:
            space_ax (plt.Axes): A matplotlib Axes for the reconstruction
            hidden_ax (plt.Axes): A matplotlib Axes for the hidden basis chart
            model (FourierFeatureMLP): The model to visualize
            num_points (int): The number of points to visualize
            colors (np.ndarray): The colors to use per point
            max_hidden (int): The maximum number of hidden units to display
        """
        x_vals = torch.linspace(self.val_x[0, 0], self.val_x[-1, 0], num_points)
        model.eval()
        model.keep_activations = True
        with torch.no_grad():
            y_vals = model(x_vals.reshape(-1, 1)).reshape(-1)
            y_vals = y_vals.cpu().numpy()

        model.keep_activations = False
        model.train()

        slope = model.layers[-1].weight.data.detach().cpu().numpy().reshape(-1)
        bias = model.layers[-1].bias.data.item()
        activation = model.activations[-1]
        activation_values = activation * slope[np.newaxis, :] + bias
        activation_range = activation_values.max(0) - activation_values.min(0)
        index = np.argsort(activation_range)[::-1]
        index = index[:max_hidden]
        cmap = plt.get_cmap("jet")
        for rank, i in enumerate(index):
            on_index = activation[:, i] > 0
            act_x = x_vals
            act_y = activation_values[:, i]
            hidden_ax.plot(act_x, act_y, color=cmap(rank / max_hidden)[:3], zorder=1, label="h{:02d}".format(i))
            act_x = x_vals[on_index]
            act_y = act_y[on_index]
            hidden_ax.scatter(act_x, act_y, color=colors[on_index], marker=".", zorder=2)

        activation_values = activation_values[activation > 0]
        hidden_ax.set_ylim(*_get_limits(activation_values))
        hidden_ax.legend(loc="upper right", ncol=2)
        space_ax.set_xlim(*self.x_lim)
        space_ax.set_ylim(*self.y_lim)
        space_ax.plot(self.val_x.numpy(), self.val_y.numpy(), "r-", label="val", zorder=1)
        space_ax.plot(self.train_x.numpy(), self.train_y.numpy(), "go", label="train", zorder=2)
        space_ax.scatter(x_vals.numpy(), y_vals, color=colors,
                         marker="P", label="pred", zorder=3)
        space_ax.legend()
