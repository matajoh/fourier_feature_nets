"""Trains a full NeRF model."""

import argparse
import json
import os

import nerf
import torch


def _parse_args():
    parser = argparse.ArgumentParser("Tiny NeRF")
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("results_dir", help="Path to output results")
    parser.add_argument("--opacity-model",
                        help="Path to the opacity model")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=400,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=100,
                        help="Number of cameras")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=float, default=50)
    parser.add_argument("--num-channels", type=int, default=256,
                        help="Number of channels in the MLP")
    parser.add_argument("--pos-freq", type=int, default=10,
                        help="Number of frequencies used for encoding")
    parser.add_argument("--pos-max-log-scale", type=float, default=9,
                        help="Value of sigma for the positional model")
    parser.add_argument("--view-freq", type=int, default=4,
                        help="Number of frequencies used for encoding")
    parser.add_argument("--view-max-log-scale", type=float, default=3,
                        help="Value of sigma for the positional model")
    parser.add_argument("--num-steps", type=int, default=50000,
                        help="Number of steps to use for training.")
    parser.add_argument("--epoch-steps", type=int, default=1000,
                        help="Interval for progress and lr decay")
    parser.add_argument("--image-interval", type=int, default=2000,
                        help="Image rendering interval")
    parser.add_argument("--crop-epochs", type=int, default=1,
                        help="Number of epochs to train on center crops")
    parser.add_argument("--seed", type=int, default=20080524,
                        help="Manual seed for the RNG")
    parser.add_argument("--use-alpha", action="store_true",
                        help="Whether to use the alpha channel as a target")
    parser.add_argument("--omit-inputs", action="store_true",
                        help="Whether to omit inputs from the input vector")
    parser.add_argument("--color-space", choices=["YCrCb", "RGB"],
                        default="RGB")
    return parser.parse_args()


def _main():
    args = _parse_args()

    torch.manual_seed(args.seed)
    model = nerf.NeRF(args.num_layers, args.num_channels,
                      args.pos_max_log_scale, args.pos_freq,
                      args.view_max_log_scale, args.view_freq,
                      [4], not args.omit_inputs)

    if args.opacity_model:
        opacity_model = nerf.load_model(args.opacity_model)
        if opacity_model is None:
            return 1

        opacity_model = opacity_model.to("cuda")
    else:
        opacity_model = None

    train_dataset = nerf.RayDataset.load(args.data_path, "train",
                                         args.num_samples, True,
                                         opacity_model, args.batch_size,
                                         args.color_space)
    val_dataset = nerf.RayDataset.load(args.data_path, "val",
                                       args.num_samples, False,
                                       opacity_model, args.batch_size,
                                       args.color_space)

    if train_dataset is None:
        return 1

    raycaster = nerf.Raycaster(model, True)
    raycaster.to("cuda")

    log = raycaster.fit(train_dataset, val_dataset, args.results_dir,
                        args.batch_size, args.learning_rate,
                        args.num_steps, args.image_interval,
                        args.crop_epochs, args.epoch_steps)

    model.save(os.path.join(args.results_dir, "nerf.pt"))

    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        json.dump(vars(args), file)
        file.write("\n\n")
        file.write("\t".join(["step", "timestamp", "psnr_train", "psnr_val"]))
        file.write("\t")
        for line in log:
            file.write("\t".join([str(val) for val in line]) + "\n")

    sp_path = os.path.join(args.results_dir, "nerf.html")
    raycaster.to_scenepic(val_dataset).save_as_html(sp_path)


if __name__ == "__main__":
    _main()
