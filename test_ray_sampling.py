import argparse

from nerf import RaySamplingDataset


def _parse_args():
    parser = argparse.ArgumentParser("Ray Sampling Tester")
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("--split", default="train",
                        help="Data split to visualize")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Number of samples to take")
    parser.add_argument("--resolution", type=int, default=50,
                        help="Ray sampling resolution")
    parser.add_argument("--num-cameras", type=int, default=10,
                        help="Number of cameras")
    parser.add_argument("--stratified", action="store_true",
                        help="Whether to randomly offset the samples")
    return parser.parse_args()


def _main():
    args = _parse_args()

    dataset = RaySamplingDataset.load(args.data_path, args.split, args.resolution,
                                      args.num_samples, args.stratified)
    if args.num_cameras < dataset.num_cameras:
        dataset = dataset.sample_cameras(args.num_cameras)

    scene = dataset.to_scenepic()
    scene.save_as_html("ray_sampling.html", "Ray Sampling")


if __name__ == "__main__":
    _main()
