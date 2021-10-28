import argparse

import nerf
import numpy as np
import scenepic as sp
import torch


def _parse_args():
    parser = argparse.ArgumentParser("Model Voxelizer")
    parser.add_argument("model_path", help="Path to the saved model")
    parser.add_argument("data_path", help="Path to the data used to train the model")
    parser.add_argument("output_path", help="Path to the output scenepic")
    parser.add_argument("--voxel-depth", type=int, default=8,
                        help="Depth of the octree to use")
    parser.add_argument("--num-cameras", type=float, default=100,
                        help="Number of cameras to use for sampling the volume")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="Number of rays to process in a batch")
    parser.add_argument("--outlier-depth", type=int, default=6,
                        help="Tree depth at which to remove outliers")
    parser.add_argument("--min-leaf-size", type=int, default=4,
                        help="Minimum number of samples in a leaf")
    return parser.parse_args()


def _main():
    args = _parse_args()

    model = nerf.load_model(args.model_path)
    dataset = nerf.RaySamplingDataset.load(args.data_path, "train", 400, 128, False)
    if args.num_cameras < dataset.num_cameras:
        dataset = dataset.sample_cameras(args.num_cameras)

    print("Sampling the model...")
    raycaster = nerf.Raycaster(model)
    raycaster.to("cuda")
    num_rays = len(dataset)
    colors = []
    positions = []
    milestone = 0
    with torch.no_grad():
        for start in range(0, num_rays, args.batch_size):
            end = min(start + args.batch_size, num_rays)

            if end >= milestone:
                print("{}%".format(int(100 * start / num_rays)))
                milestone += num_rays / 20

            index = list(range(start, end))
            rays = dataset[list(range(start, end))]
            color, alpha, depth = raycaster.render(rays.to("cuda"), True)
            valid = (alpha > 0.3).cpu()
            colors.append(color[valid].cpu().numpy())
            starts = dataset.starts[index]
            dirs = dataset.directions[index]
            position = starts + dirs * depth.cpu().unsqueeze(-1)
            positions.append(position[valid].cpu().numpy())

    positions = np.concatenate(positions)
    colors = np.concatenate(colors)

    voxels, leaf_colors = nerf.OcTree.build_from_samples(positions, colors,
                                                         args.voxel_depth,
                                                         args.outlier_depth,
                                                         args.min_leaf_size)
    leaf_centers = voxels.leaf_centers()
    leaf_depths = voxels.leaf_depths()

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=800, height=800)
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

    scene.save_as_html(args.output_path)


if __name__ == "__main__":
    _main()
