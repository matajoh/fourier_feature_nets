"""Script which creates an orbit video of a trained model."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import cv2
import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp
import torch


def _parse_args():
    parser = ArgumentParser("Orbit Video Maker",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("resolution", type=int, help="Resolution of the video")
    parser.add_argument("output_dir", help="Output directory for the images")
    parser.add_argument("--opacity-model",
                        help="Optional path to an opacity model.")
    parser.add_argument("--distance", type=float, default=4,
                        help="Distance of the camera")
    parser.add_argument("--fov-y-degrees", type=float, default=40,
                        help="Camera field of view in degrees")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of frames in the video")
    parser.add_argument("--up-dir", default="y+",
                        choices=list(VECTORS.keys()),
                        help="The direction that is 'up'")
    parser.add_argument("--forward-dir", default="z-",
                        choices=list(VECTORS.keys()),
                        help="The direction that is 'forward'")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples per ray.")
    parser.add_argument("--alpha-thresh", type=float, default=0.3,
                        help="Alpha threshold below which pixels are omitted.")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for rendering.")
    parser.add_argument("--device", default="cuda",
                        help="Pytorch compute device")
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

    up_dir = VECTORS[args.up_dir]
    forward_dir = VECTORS[args.forward_dir]
    orbit_cameras = ffn.orbit(up_dir, forward_dir, args.num_frames,
                              args.fov_y_degrees,
                              ffn.Resolution(args.resolution, args.resolution),
                              args.distance)

    bounds_transform = sp.Transforms.scale(2)

    model = ffn.load_model(args.model_path)
    model = model.to(args.device)

    if args.opacity_model:
        opacity_model = ffn.load_model(args.opacity_model)
        opacity_model.to(args.device)
    else:
        opacity_model = model

    raycaster = ffn.Raycaster(model, isinstance(model, ffn.NeRF))
    sampler = ffn.RaySampler(bounds_transform, orbit_cameras,
                             args.num_samples, False,
                             opacity_model, args.batch_size)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        for frame in range(args.num_frames):
            print(frame, "/", args.num_frames)
            image = raycaster.render_image(sampler, frame, args.batch_size)
            path = os.path.join(args.output_dir,
                                "frame_{:04d}.png".format(frame))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)


if __name__ == "__main__":
    _main()
