"""Trains a Tiny NeRF model (only positional data)."""

import argparse
import json
import os

import nerf
import torch


def _parse_args():
    parser = argparse.ArgumentParser("Tiny NeRF")
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("nerf_model", choices=["mlp", "basic",
                                               "positional", "gaussian"])
    parser.add_argument("results_dir", help="Path to output results")
    parser.add_argument("--opacity-model",
                        help="Path to the opacity model")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-channels", type=int, default=256,
                        help="Number of channels in the MLP")
    parser.add_argument("--embedding-size", type=int, default=256,
                        help="Embedding size used for encoding")
    parser.add_argument("--pos-max-log-scale", type=float, default=1.27,
                        help="Max log scale for the positional encoding")
    parser.add_argument("--gauss-sigma", type=float, default=6.05,
                        help="Standard deviation for the gaussian encoding")
    parser.add_argument("--num-steps", type=int, default=50000,
                        help="Number of steps to use for training.")
    parser.add_argument("--report-interval", type=int, default=1000,
                        help="Reporting interval for validation/logging")
    parser.add_argument("--crop-epochs", type=int, default=1,
                        help="Number of epochs to train on center crops")
    parser.add_argument("--seed", type=int, default=20080524,
                        help="Manual seed for the RNG")
    parser.add_argument("--use-alpha", action="store_true",
                        help="Whether to use the alpha channel as a target")
    return parser.parse_args()


def _main():
    args = _parse_args()

    torch.manual_seed(args.seed)
    if args.nerf_model == "mlp":
        model = nerf.MLP(3, 4, num_channels=args.num_channels)
    elif args.nerf_model == "basic":
        model = nerf.BasicFourierMLP(3, 4, num_channels=args.num_channels)
    elif args.nerf_model == "positional":
        model = nerf.PositionalFourierMLP(3, 4,
                                          max_log_scale=args.pos_max_log_scale,
                                          num_channels=args.num_channels,
                                          embedding_size=args.embedding_size)
    elif args.nerf_model == "gaussian":
        model = nerf.GaussianFourierMLP(3, 4,
                                        sigma=args.gauss_sigma,
                                        num_channels=args.num_channels,
                                        embedding_size=args.embedding_size)

    if args.opacity_model:
        opacity_model = nerf.load_model(args.opacity_model)
        if opacity_model is None:
            return 1

        opacity_model = opacity_model.to("cuda")
    else:
        opacity_model = None

    train_dataset = nerf.RaySamplingDataset.load(args.data_path, "train",
                                                 args.num_samples, True,
                                                 opacity_model,
                                                 args.batch_size)
    val_dataset = nerf.RaySamplingDataset.load(args.data_path, "val",
                                               args.num_samples, False,
                                               opacity_model,
                                               args.batch_size)

    if train_dataset is None:
        return 1

    raycaster = nerf.Raycaster(model)
    raycaster.to("cuda")

    log = raycaster.fit(train_dataset, val_dataset, args.results_dir,
                        args.batch_size, args.learning_rate,
                        args.num_steps, args.report_interval,
                        args.crop_epochs)

    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        json.dump(vars(args), file)
        file.write("\n\n")
        file.write("\t".join(["step", "timestamp", "psnr_train", "psnr_val"]))
        file.write("\t")
        for line in log:
            file.write("\t".join([str(val) for val in line]) + "\n")

    model.save(os.path.join(args.results_dir, "tiny_nerf.pt"))
    sp_path = os.path.join(args.results_dir, "tiny_nerf.html")
    raycaster.to_scenepic(val_dataset).save_as_html(sp_path)


if __name__ == "__main__":
    _main()
