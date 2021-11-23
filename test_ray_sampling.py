"""Produces a visualization of a ray sampling dataset."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from fourier_feature_nets import load_model, RayDataset


def _parse_args():
    parser = ArgumentParser("Ray Sampling Tester",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", help="Path to the data NPZ")
    parser.add_argument("output_path", help="Path to the output scenepic")
    parser.add_argument("--mode",
                        choices=["full", "sparse", "dilate", "center"],
                        default="full", help="The dataset sampling mode")
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
    parser.add_argument("--opacity-model",
                        help="Path to a model to use to predict opacity")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size to use when quering the opacity model")
    return parser.parse_args()


def _main():
    args = _parse_args()

    if args.opacity_model:
        model = load_model(args.opacity_model)
        if model is None:
            return 1

        model = model.to("cuda")
    else:
        model = None

    dataset = RayDataset.load(args.data_path, args.split,
                              args.num_samples, True,
                              args.stratified, model,
                              args.batch_size)
    if dataset is None:
        return 1

    if args.num_cameras and args.num_cameras < dataset.num_cameras:
        dataset = dataset.sample_cameras(args.num_cameras,
                                         args.num_samples,
                                         args.stratified)

    if args.mode == "sparse":
        dataset.mode = RayDataset.Mode.Sparse
    elif args.mode == "center":
        dataset.mode = RayDataset.Mode.Center
    elif args.mode == "dilate":
        dataset.mode = RayDataset.Mode.Dilate

    scene = dataset.to_scenepic(args.resolution)
    scene.save_as_html(args.output_path, "Ray Sampling")


if __name__ == "__main__":
    _main()
