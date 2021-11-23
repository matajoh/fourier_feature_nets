import argparse

import cv2
import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("mp4_path")
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--up-dir", default="0,1,0")
    parser.add_argument("--forward-dir", default="0,0,-1")
    parser.add_argument("--framerate", type=float, default=10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--distance", type=float, default=3)
    return parser.parse_args()


def _main():
    args = _parse_args()
    up_dir = np.array([float(x) for x in args.up_dir.split(",")], np.float32)
    forward_dir = np.array([float(x) for x in args.forward_dir.split(",")],
                           np.float32)

    data = np.load(args.data_path)
    images = data["images"]
    height, width = images.shape[1:3]
    src_resolution = ffn.Resolution(width, height)
    resolution = src_resolution.scale_to_height(args.resolution).square()
    train_count = data["split_counts"][0]
    train_extrinsics = data["extrinsics"][:train_count]
    data_positions = np.stack([extrinsics[:3, 3]
                               for extrinsics in train_extrinsics])

    orbit_cameras = ffn.orbit(up_dir, forward_dir,
                               args.num_frames, 40, resolution, args.distance)

    orbit_positions = np.stack([cam.position for cam in orbit_cameras])

    orbit_positions = orbit_positions.reshape(args.num_frames, 1, 3)
    data_positions = data_positions.reshape(1, -1, 3)
    distances = np.square(orbit_positions - data_positions).sum(-1)
    gt_index = distances.argmin(-1)

    with sp.VideoWriter(args.mp4_path, resolution,
                        rgb=True, framerate=args.framerate) as writer:
        for i in gt_index:
            if src_resolution.width != src_resolution.height:
                start = (src_resolution.width - src_resolution.height) // 2
                end = start + src_resolution.height
                image = images[i, :, start:end]
            else:
                image = images[i]

            writer.frame[:] = cv2.resize(image, resolution)[..., :3]
            writer.write_frame()


if __name__ == "__main__":
    _main()
