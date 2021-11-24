"""Interprets a volumetric model function as voxels."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp
import torch


def _parse_args():
    parser = ArgumentParser("Model Voxelizer",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_path", help="Path to the saved model")
    parser.add_argument("data_path", help="Path to the data used to train the model")
    parser.add_argument("output_path", help="Path to the output octree")
    parser.add_argument("--scenepic-path")
    parser.add_argument("--voxel-depth", type=int, default=8,
                        help="Depth of the octree to use")
    parser.add_argument("--num-cameras", type=float, default=100,
                        help="Number of cameras to use for sampling the volume")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Number of rays to process in a batch")
    parser.add_argument("--min-leaf-size", type=int, default=4,
                        help="Minimum number of samples in a leaf")
    parser.add_argument("--alpha-threshold", type=float, default=0.3,
                        help="Threshold to use when filtering samples")
    parser.add_argument("--opacity-model-path",
                        help="Path to an optional opacity model")
    parser.add_argument("--device", default="cuda",
                        help="Pytorch compute device")
    return parser.parse_args()


def _main():
    args = _parse_args()

    model = ffn.load_model(args.model_path)
    if model is None:
        return 1

    if args.opacity_model_path:
        opacity_model = ffn.load_model(args.opacity_model_path)
        opacity_model = opacity_model.to(args.device)
    else:
        opacity_model = None

    dataset = ffn.RayDataset.load(args.data_path, "train", 400, 128, False,
                                  opacity_model)
    if dataset is None:
        return 1

    if args.num_cameras < dataset.num_cameras:
        dataset = dataset.sample_cameras(args.num_cameras,
                                         dataset.num_samples,
                                         False)

    sampler = dataset.sampler
    model = model.to(args.device)
    raycaster = ffn.Raycaster(model)
    raycaster.to(args.device)
    num_rays = len(sampler)
    colors = []
    positions = []
    bar = ffn.ETABar("Sampling the model", max=num_rays)
    with torch.no_grad():
        for start in range(0, num_rays, args.batch_size):
            end = min(start + args.batch_size, num_rays)
            index = list(range(start, end))
            rays = sampler[list(range(start, end))]
            color, alpha, depth = raycaster.render(rays.to(args.device), True)
            valid = (alpha > args.alpha_threshold).cpu()
            colors.append(color[valid].cpu().numpy())
            starts = sampler.starts[index]
            dirs = sampler.directions[index]
            position = starts + dirs * depth.cpu().unsqueeze(-1)
            positions.append(position[valid].cpu().numpy())
            bar.next(end - start)

    bar.finish()
    positions = np.concatenate(positions)
    colors = np.concatenate(colors)

    print(len(positions), "points in cloud")
    voxels = ffn.OcTree.build_from_samples(positions,
                                           args.voxel_depth,
                                           args.min_leaf_size,
                                           colors)
    voxels.save(args.output_path)

    if args.scenepic_path:
        leaf_centers = voxels.leaf_centers()
        leaf_depths = voxels.leaf_depths()
        leaf_colors = voxels.leaf_data()

        scene = sp.Scene()
        canvas = scene.create_canvas_3d(width=800, height=800)
        canvas.shading = sp.Shading(sp.Colors.White)
        frame = canvas.create_frame()

        depths = np.unique(leaf_depths)
        for depth in depths:
            mesh = scene.create_mesh()
            transform = sp.Transforms.scale(pow(2., 1-depth) * voxels.scale)
            mesh.add_cube(sp.Colors.White, transform=transform)
            depth_centers = leaf_centers[leaf_depths == depth]
            depth_colors = leaf_colors[leaf_depths == depth]
            mesh.enable_instancing(depth_centers, colors=depth_colors)
            frame.add_mesh(mesh)

        scene.save_as_html(args.scenepic_path)


if __name__ == "__main__":
    _main()
