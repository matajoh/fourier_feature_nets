import argparse

from nerf import RaySamplingDataset


def _parse_args():
    parser = argparse.ArgumentParser("Ray Sampling Tester")
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("--voxels-dir", help="Path to the voxels directory")
    parser.add_argument("--path-length", type=int, default=128,
                        help="Number of voxels to intersect")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=10,
                        help="Number of cameras")
    parser.add_argument("--stratified", action="store_true",
                        help="Whether to randomly offset the samples")
    return parser.parse_args()


def _main():
    args = _parse_args()

    dataset = RaySamplingDataset.load(args.data_path, "test", args.resolution,
                                      args.num_samples, args.stratified)
    dataset.to_scenepic().save_as_html("ray_sampling.html")


if __name__ == "__main__":
    _main()
