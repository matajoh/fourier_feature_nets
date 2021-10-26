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
    parser.add_argument("--voxels-dir",
                        help="Path to the voxels directory")
    parser.add_argument("--path-length", type=int, default=128,
                        help="Number of voxels to intersect")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=400,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=100,
                        help="Number of cameras")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-channels", type=int, default=256,
                        help="Number of channels in the MLP")
    parser.add_argument("--num-frequencies", type=int, default=256,
                        help="Number of frequencies used for encoding")
    parser.add_argument("--pos-sigma", type=float, default=1.27,
                        help="Value of sigma for the positional model")
    parser.add_argument("--gauss-sigma", type=float, default=6.05,
                        help="Value of sigma for the gaussian model")
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
        model = nerf.PositionalFourierMLP(3, 4, sigma=args.pos_sigma,
                                          num_channels=args.num_channels,
                                          num_frequencies=args.num_frequencies)
    elif args.nerf_model == "gaussian":
        model = nerf.GaussianFourierMLP(3, 4, sigma=args.gauss_sigma,
                                        num_channels=args.num_channels,
                                        num_frequencies=args.num_frequencies)

    data_path = args.data_path
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "data", data_path)
        if not os.path.exists(data_path):
            dataset_name = os.path.basename(data_path)[:-4]
            success = nerf.RaySamplingDataset.download(dataset_name, data_path)
            if not success:
                print("Unable to download dataset", dataset_name)
                return 1

    train_dataset = nerf.RaySamplingDataset.load(data_path, "train",
                                                 args.resolution,
                                                 args.num_samples, True)
    val_dataset = nerf.RaySamplingDataset.load(data_path, "val",
                                               args.resolution,
                                               args.num_samples, False)

    raycaster = nerf.Raycaster(train_dataset, val_dataset, model,
                               args.results_dir)
    raycaster.to("cuda")

    log = raycaster.fit(args.batch_size, args.learning_rate,
                        args.num_steps, args.report_interval,
                        args.crop_epochs)

    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        json.dump(vars(args), file)
        file.write("\n\n")
        file.write("\t".join(["step", "timestamp", "psnr_train", "psnr_val"]))
        file.write("\t")
        for line in log:
            file.write("\t".join([str(val) for val in line]) + "\n")

    sp_path = os.path.join(args.results_dir, "tiny_nerf.html")
    raycaster.to_scenepic().save_as_html(sp_path)
    model.save(os.path.join(args.results_dir, "tiny_nerf.model"))


if __name__ == "__main__":
    _main()
