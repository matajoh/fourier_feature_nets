"""Trains a voxelized volumetric representation from images."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json
import os

import fourier_feature_nets as ffn
import numpy as np
import torch


def _parse_args():
    parser = ArgumentParser("Voxel Training Script",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("side", type=int, help="One side of the voxel volume")
    parser.add_argument("results_dir", help="Path to output results")
    parser.add_argument("--mode",
                        choices=["rgba", "rgb", "dilate"],
                        default="rgba")
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of samples to take")
    parser.add_argument("--num-cameras", type=int, default=100,
                        help="Number of cameras")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of steps to use for training.")
    parser.add_argument("--report-interval", type=int, default=1000,
                        help="Interval for progress")
    parser.add_argument("--image-interval", type=int, default=2000,
                        help="Image rendering interval")
    parser.add_argument("--seed", type=int, default=20080524,
                        help="Manual seed for the RNG")
    parser.add_argument("--decay-rate", type=float, default=0.9,
                        help="Rate at which the learning rate decays")
    parser.add_argument("--decay-steps", type=int, default=25000,
                        help="Interval over which the learning rate decays.")
    parser.add_argument("--color-space", choices=["YCrCb", "RGB"],
                        default="RGB",
                        help="Color space to use for training")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of frames in the training video orbit.")
    parser.add_argument("--device", default="cuda",
                        help="Pytorch compute device")
    parser.add_argument("--anneal-start", type=float, default=0.2,
                        help="Starting value for the sample space annealing.")
    parser.add_argument("--num-anneal-steps", type=int, default=0,
                        help=("Steps over which to anneal sampling to the full"
                              "range of volume intersection."))
    parser.add_argument("--up-dir", default="y+",
                        choices=list(VECTORS.keys()),
                        help="The direction that is 'up'")
    parser.add_argument("--forward-dir", default="z-",
                        choices=list(VECTORS.keys()),
                        help="The direction that is 'forward'")
    parser.add_argument("--num-patch-cameras", type=int, default=100,
                        help="Number of cameras to use for patch sampling")
    parser.add_argument("--num-patches", type=int, default=256,
                        help="Number of patches per patch camera")
    parser.add_argument("--patch-size", type=int, default=8,
                        help="size of the patches")

    return parser.parse_args()


VECTORS = {
    "x+": np.array([1, 0, 0], np.float32),
    "x-": np.array([-1, 0, 0], np.float32),
    "y+": np.array([0, 1, 0], np.float32),
    "y-": np.array([0, -1, 0], np.float32),
    "z+": np.array([0, 0, 1], np.float32),
    "z-": np.array([0, 0, -1], np.float32),
}


def _main():
    args = _parse_args()

    torch.manual_seed(args.seed)

    include_alpha = args.mode == "rgba"
    dataset = ffn.ImageDataset.load(args.data_path, "train",
                                    args.num_samples, include_alpha,
                                    True, color_space=args.color_space,
                                    anneal_start=args.anneal_start,
                                    num_anneal_steps=args.num_anneal_steps)

    patch_cameras = ffn.hemisphere(VECTORS[args.up_dir],
                                   VECTORS[args.forward_dir],
                                   100, dataset.cameras[0].fov_y_degrees,
                                   dataset.cameras[0].resolution, 4)
    patches = ffn.PatchesDataset("patches", dataset.sampler.bounds,
                                 patch_cameras, args.num_samples,
                                 args.num_patches, args.patch_size, True,
                                 color_space=args.color_space,
                                 anneal_start=args.anneal_start,
                                 num_anneal_steps=args.num_anneal_steps)

    patches.to_scenepic().save_as_html("patches.html")

    assert len(dataset) > 2 * args.num_images

    dataset = dataset.sample_cameras(args.num_images * 2, args.num_samples,
                                     True)

    index = list(range(2 * args.num_images))

    train_dataset = dataset.subset(index[:args.num_images],
                                   args.num_samples, True)
    val_dataset = dataset.subset(index[args.num_images:],
                                 args.num_samples, False)

    if train_dataset is None:
        return 1

    visualizers = [ffn.ComparisonVisualizer(
        args.results_dir,
        args.num_steps,
        args.num_frames,
        train_dataset,
        val_dataset
    )]

    if args.mode == "dilate":
        train_dataset.mode = ffn.RayDataset.Mode.Dilate

    scale = 2 / train_dataset.sampler.bounds[0, 0]
    model = ffn.Voxels(args.side, scale)

    raycaster = ffn.Raycaster(model)
    raycaster.to(args.device)

    log = raycaster.fit(train_dataset, val_dataset, args.batch_size,
                        args.learning_rate, args.num_steps, 0,
                        args.report_interval, args.decay_rate, args.decay_steps,
                        0.0, visualizers)

    model.save(os.path.join(args.results_dir, "voxels.pt"))
    with open(os.path.join(args.results_dir, "log.txt"), "w") as file:
        json.dump(vars(args), file)
        file.write("\n\n")
        file.write("\t".join(["step", "timestamp", "psnr_train", "psnr_val"]))
        file.write("\n")
        for entry in log:
            file.write("\t".join([str(val) for val in [
                entry.step, entry.timestamp, entry.train_psnr, entry.val_psnr
            ]]) + "\n")

    sp_path = os.path.join(args.results_dir, "voxels.html")
    raycaster.to_scenepic(val_dataset).save_as_html(sp_path)


if __name__ == "__main__":
    _main()
