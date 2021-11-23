"""Interprets a volumetric model function as voxels."""

import argparse

import fourier_feature_nets as ffn
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser("Model Voxelizer")
    parser.add_argument("mesh_path", help="Path to the OBJ file")
    parser.add_argument("output_path", help="Path to the output NPZ")
    parser.add_argument("--voxel-depth", type=int, default=8,
                        help="Depth of the octree to use")
    parser.add_argument("--min-leaf-size", type=int, default=4,
                        help="Minimum number of samples in a leaf")
    parser.add_argument("--up-dir", default="0,1,0")
    return parser.parse_args()


def _main():
    args = _parse_args()

    up_dir = [float(val) for val in args.up_dir.split(",")]
    up_dir = np.array(up_dir, np.float32)

    print("Building the octree")
    voxels = ffn.OcTree.build_from_mesh(args.mesh_path, args.voxel_depth,
                                         args.min_leaf_size, up_dir)
    voxels.save(args.output_path)


if __name__ == "__main__":
    _main()
