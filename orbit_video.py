"""Script which creates an orbit video of a trained model."""

import argparse

import nerf
import numpy as np
import scenepic as sp


def _parse_args():
    parser = argparse.ArgumentParser("Orbit Video Maker")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("resolution", type=int, help="Resolution of the video")
    parser.add_argument("output_path", help="Path to the output MP4")
    parser.add_argument("--distance", type=int, default=4,
                        help="Distance of the camera")
    parser.add_argument("--fov-y-degrees", type=float, default=40,
                        help="Camera field of view in degrees")
    parser.add_argument("--num-frames", type=int, default=120,
                        help="Number of frames in the video")
    parser.add_argument("--up-dir", default="y+", 
                        choices=["x+","x-","y+","y-","z+","z-"],
                        help="The direction that is 'up'")
    return parser.parse_args()


UP_VECTORS = {
    "x+": np.array([1, 0, 0], np.float32),
    "x-": np.array([-1, 0, 0], np.float32),
    "y+": np.array([0, 1, 0], np.float32),
    "y-": np.array([0, -1, 0], np.float32),
    "z+": np.array([0, 0, 1], np.float32),
    "z-": np.array([0, 0, -1], np.float32),
}


def _main():
    args = _parse_args()

    up_dir = UP_VECTORS[args.up_dir]

    model = nerf.load_model(args.model_path)
    azimuth = np.linspace(0, 2*np.pi, args.num_frames)
    altitude = np.abs(azimuth - np.pi) / 3
    
    
