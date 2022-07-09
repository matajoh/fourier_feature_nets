"""Trains a Tiny NeRF model (only positional data)."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
import os

import fourier_feature_nets as ffn
import torch


def _parse_args():
    parser = ArgumentParser("Tiny NeRF Training Script",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("nerf_model", choices=["mlp", "basic",
                                               "positional", "gaussian"])
    parser.add_argument("results_dir", help="Path to output results")
    parser.add_argument("--mode",
                        choices=["rgba", "rgb", "dilate"],
                        default="rgba")
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
    parser.add_argument("--pos-max-log-scale", type=float, default=5.5,
                        help="Max log scale for the positional encoding")
    parser.add_argument("--gauss-sigma", type=float, default=6.05,
                        help="Standard deviation for the gaussian encoding")
    parser.add_argument("--num-steps", type=int, default=50000,
                        help="Number of steps to use for training.")
    parser.add_argument("--report-interval", type=int, default=1000,
                        help="Interval for progress reports")
    parser.add_argument("--image-interval", type=int, default=2000,
                        help="Image rendering interval")
    parser.add_argument("--crop-steps", type=int, default=1000,
                        help="Number of epochs to train on center crops")
    parser.add_argument("--seed", type=int, default=20080524,
                        help="Manual seed for the RNG")
    parser.add_argument("--decay-rate", type=float, default=0.1,
                        help="Rate at which the learning rate decays")
    parser.add_argument("--decay-steps", type=int, default=25000,
                        help="Interval over which the learning rate decays.")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Regularizer term for the weights.")
    parser.add_argument("--make-video", action="store_true",
                        help="Whether to render frames for a training video.")
    parser.add_argument("--make-activations", action="store_true",
                        help="Whether to render frames for an activations video.")
    parser.add_argument("--color-space", choices=["YCrCb", "RGB"],
                        default="RGB",
                        help="Color space to use during training.")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of frames in the training video orbit.")
    parser.add_argument("--device", default="cuda",
                        help="Pytorch compute device")
    parser.add_argument("--anneal-start", type=float, default=0.2,
                        help="Starting value for the sample space annealing.")
    parser.add_argument("--num-anneal-steps", type=int, default=2000,
                        help=("Steps over which to anneal sampling to the full"
                              "range of volume intersection."))

    return parser.parse_args()


def _main():
    args = _parse_args()

    torch.manual_seed(args.seed)
    if args.nerf_model == "mlp":
        model = ffn.MLP(3, 4, num_channels=args.num_channels)
    elif args.nerf_model == "basic":
        model = ffn.BasicFourierMLP(3, 4, num_channels=args.num_channels)
    elif args.nerf_model == "positional":
        model = ffn.PositionalFourierMLP(3, 4,
                                         max_log_scale=args.pos_max_log_scale,
                                         num_channels=args.num_channels,
                                         embedding_size=args.embedding_size)
    elif args.nerf_model == "gaussian":
        model = ffn.GaussianFourierMLP(3, 4,
                                       sigma=args.gauss_sigma,
                                       num_channels=args.num_channels,
                                       embedding_size=args.embedding_size)

    if args.opacity_model:
        opacity_model = ffn.load_model(args.opacity_model)
        if opacity_model is None:
            return 1

        opacity_model = opacity_model.to(args.device)
    else:
        opacity_model = None

    include_alpha = args.mode == "rgba"
    train_dataset = ffn.ImageDataset.load(args.data_path, "train",
                                          args.num_samples, include_alpha,
                                          True, opacity_model,
                                          args.batch_size, args.color_space,
                                          anneal_start=args.anneal_start,
                                          num_anneal_steps=args.num_anneal_steps)
    val_dataset = ffn.ImageDataset.load(args.data_path, "val",
                                        args.num_samples, include_alpha,
                                        False, opacity_model,
                                        args.batch_size, args.color_space)

    if train_dataset is None:
        return 1

    visualizers = []
    if args.make_video:
        resolution = train_dataset.cameras[0].resolution
        visualizers.append(ffn.OrbitVideoVisualizer(
            args.results_dir,
            args.num_steps,
            resolution,
            args.num_frames,
            args.num_samples,
            args.color_space
        ))
    else:
        visualizers.append(ffn.EvaluationVisualizer(
            args.results_dir,
            train_dataset,
            args.image_interval
        ))
        visualizers.append(ffn.EvaluationVisualizer(
            args.results_dir,
            val_dataset,
            args.image_interval
        ))

    if args.make_activations:
        resolution = train_dataset.cameras[0].resolution
        visualizers.append(ffn.ActivationVisualizer(
            args.results_dir,
            args.num_steps,
            resolution,
            args.num_frames,
            args.num_samples,
            args.color_space
        ))

    if args.mode == "dilate":
        train_dataset.mode = ffn.RayDataset.Mode.Dilate

    raycaster = ffn.Raycaster(model.to(args.device))

    log = raycaster.fit(train_dataset, val_dataset,
                        args.batch_size, args.learning_rate,
                        args.num_steps, args.crop_steps, args.report_interval,
                        args.decay_rate, args.decay_steps,
                        args.weight_decay, visualizers)

    model.save(os.path.join(args.results_dir, "tiny_nerf.pt"))

    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        json.dump(vars(args), file)
        file.write("\n\n")
        file.write("\t".join(["step", "timestamp", "psnr_train", "psnr_val"]))
        file.write("\t")
        for entry in log:
            file.write("\t".join([str(val) for val in [
                entry.step, entry.timestamp, entry.train_psnr, entry.val_psnr
            ]]) + "\n")

    sp_path = os.path.join(args.results_dir, "tiny_nerf.html")
    raycaster.to_scenepic(val_dataset).save_as_html(sp_path)


if __name__ == "__main__":
    _main()
