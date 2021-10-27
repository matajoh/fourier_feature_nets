import argparse

import nerf
import numpy as np
import scenepic as sp
import torch


def _parse_args():
    parser = argparse.ArgumentParser("Model Voxelizer")
    parser.add_argument("model_path", help="Path to the saved model")
    parser.add_argument("type", choices=["fourier", "nerf", "voxels"],
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
    leaf_centers = voxels.leaf_centers()
    leaf_depths = voxels.leaf_depths()
    with torch.no_grad():
        leaf_colors = model(torch.from_numpy(leaf_centers))
        leaf_colors, _ = torch.split(leaf_colors, [3, 1], -1)
        leaf_colors = torch.sigmoid(leaf_colors)
        leaf_colors = leaf_colors.cpu().numpy()

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=800, height=800)
    frame = canvas.create_frame()

    depths = np.unique(leaf_depths)
    for depth in depths:
        positions = leaf_centers[leaf_depths == depth]
        colors = leaf_colors[leaf_depths == depth]
        transform = sp.Transforms.scale(pow(2, 1-depth) * args.scale)
        mesh = scene.create_mesh()
        mesh.add_cube(sp.Colors.White, transform=transform)
        mesh.enable_instancing(positions, colors=colors)
        frame.add_mesh(mesh)

    scene.save_as_html(args.output_path)


if __name__ == "__main__":
    _main()
