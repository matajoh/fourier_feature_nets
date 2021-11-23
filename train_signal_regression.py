"""Script to train a model to perform 1-D signal regression."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
from typing import NamedTuple

import cv2
from fourier_feature_nets import FourierFeatureMLP, SignalDataset
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import scenepic as sp
import torch


def _multifreq(x):
    return 2 + np.sin(x*np.pi) + 0.5*np.sin(2*x*np.pi) - 0.2*np.cos(5*x*np.pi)


def _triangle(x):
    section_length = 0.5
    section0 = x < section_length
    section1 = (x >= section_length) & (x < 2 * section_length)
    section2 = (x >= 2 * section_length) & (x < 3 * section_length)
    section3 = x >= 3 * section_length
    output = np.zeros_like(x)
    output[section0] = x[section0]
    output[section1] = 2 * section_length - x[section1]
    output[section2] = x[section2] - 2 * section_length
    output[section3] = 4 * section_length - x[section3]
    return output


def _sawtooth(x):
    section_length = 0.5
    section0 = x < section_length
    section1 = (x >= section_length) & (x < 2 * section_length)
    section2 = (x >= 2 * section_length) & (x < 3 * section_length)
    section3 = x >= 3 * section_length
    output = np.zeros_like(x)
    output[section0] = x[section0]
    output[section1] = x[section1] - section_length
    output[section2] = x[section2] - 2 * section_length
    output[section3] = x[section3] - 3 * section_length
    return output


def _parse_args():
    parser = ArgumentParser("1-D Signal Regression",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("signal", choices=["multifreq", "sawtooth", "triangle"],
                        help="Signal to use for the dataset.")
    parser.add_argument("results_dir", help="Output directory")
    parser.add_argument("--num-channels", type=int, default=64,
                        help="Number of channels in the MLP")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of layers in the MLP")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Number of samples to use for training.")
    parser.add_argument("--sample-rate", type=int, default=8,
                        help="The rate at which training samples occur.")
    parser.add_argument("--num_plot", type=int, default=48,
                        help="The number of points to plot in the display.")
    parser.add_argument("--max-hidden", type=int, default=10,
                        help="Maximum number of hidden units to display.")
    parser.add_argument("--fourier", action="store_true",
                        help="Whether to use fourier features.")
    parser.add_argument("--resolution", default="1280x720",
                        help="Resolution of the display")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of training steps.")
    parser.add_argument("--make-video", action="store_true",
                        help="Whether to record an MP4")
    parser.add_argument("--framerate", type=int, default=5,
                        help="Framerate for the output video")
    parser.add_argument("--no-plot", action="store_true",
                        help="Whether to run headless (no display)")
    return parser.parse_args()


def _loss(model: FourierFeatureMLP,
          x: torch.Tensor,
          target_y: torch.Tensor) -> torch.Tensor:
    output_y = model(x)
    return (output_y - target_y).square().mean()


def _validate(model: FourierFeatureMLP,
              dataset: SignalDataset) -> float:
    model.eval()
    with torch.no_grad():
        loss = _loss(model, dataset.val_x, dataset.val_y)

    model.train()
    return loss.item()


LogEntry = NamedTuple("LogEntry", [("step", int),
                                   ("train_loss", float),
                                   ("val_loss", float)])


def _main():
    args = _parse_args()
    if args.signal == "multifreq":
        function = _multifreq
    elif args.signal == "sawtooth":
        function = _sawtooth
    elif args.signal == "triangle":
        function = _triangle

    dataset = SignalDataset.create(function, args.num_samples, args.sample_rate)

    if args.fourier:
        b_values = np.arange(1, args.num_samples // 2 + 1).astype(np.float32)
        b_values = torch.from_numpy(b_values).reshape(1, -1)
        a_values = 1 / np.arange(1, args.num_samples // 2 + 1).astype(np.float32)
        a_values = torch.from_numpy(a_values)
        learning_rate = 5e-4
    else:
        a_values = b_values = None
        learning_rate = 5e-4

    model = FourierFeatureMLP(1, 1, a_values, b_values,
                              [args.num_channels] * args.num_layers)
    model.layers[-1].bias.data = dataset.train_y.mean()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    width, height = [int(val) for val in args.resolution.split("x")]
    if not args.no_plot:
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, args.num_plot))[..., :3]
        hidden_ax = fig.add_subplot(121)
        space_ax = fig.add_subplot(122)
    else:
        fig = canvas = colors = hidden_ax = space_ax = None

    optim = torch.optim.Adam(model.parameters(), learning_rate,
                             weight_decay=1e-3)
    if args.make_video:
        mp4_path = os.path.join(args.results_dir, "training.mp4")
        writer = sp.VideoWriter(mp4_path, (width, height), quality=1,
                                framerate=args.framerate)
        writer.start()
        args.no_plot = False
    else:
        writer = None

    log = []
    for step in range(args.num_steps + 1):
        optim.zero_grad()
        loss = _loss(model, dataset.train_x, dataset.train_y)
        loss.backward()
        optim.step()
        if step % 50 == 0 or step == args.num_steps:
            val_loss = _validate(model, dataset)
            if not args.no_plot:
                space_ax.cla()
                hidden_ax.cla()
                hidden_ax.set_title("Hidden Layer Basis")
                title = "{}MLP {}x{} {:.3f}@{:05d}".format(
                    "Fourier " if args.fourier else "",
                    args.num_layers, args.num_channels,
                    val_loss, step)
                space_ax.set_title(title)
                dataset.plot(space_ax, hidden_ax, model, args.num_plot, colors,
                             args.max_hidden)
                fig.tight_layout()
                canvas.draw()
                buf = canvas.buffer_rgba()
                pixels = np.asarray(buf)[..., :3]
                pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                cv2.imshow("progress", pixels)

                if writer:
                    writer.frame[:] = pixels
                    writer.write_frame()

            print(step, "train:", loss.item(), "val:", val_loss)
            log.append(LogEntry(step, loss.item(), val_loss))

        cv2.waitKey(1)

    if writer:
        writer.stop()

    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        file.write("step\ttrain_loss\tval_loss\n")
        for i, train_loss, val_loss in log:
            file.write("{}\t{}\t{}\n".format(i, train_loss, val_loss))


if __name__ == "__main__":
    _main()
