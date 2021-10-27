import argparse

import nerf
import scenepic as sp


def _parse_args():
    parser = argparse.ArgumentParser("Model Voxelizer")
    parser.add_argument("model_path", help="Path to the saved model")
    parser.add_argument("type", choices=["mlp", "positional", "gaussian",
                                         "nerf", "voxels"],
                        help="Type of the model to load")
    parser.add_argument("output_path", help="Path to the output scenepic")
    parser.add_argument("--num-voxels", type=int, default=10**5,
                        help="Number of voxels to use")
    parser.add_argument("--scale", type=float, default=2,
                        help="Scale of the voxel volume")
    parser.add_argument("--opacity-threshold", type=float, default=0.9, 
                        help="Threshold used to determine which voxels are pruned from the tree")
    return parser.parse_args()


def _main():
    args = _parse_args()

    model = nerf.Voxels.load(args.model_path)
    voxels = nerf.OcTree.build_from_model(model, args.num_voxels,
                                          args.scale, args.opacity_threshold)


if __name__ == "__main__":
    _main()
