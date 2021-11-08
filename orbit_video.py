"""Script which creates an orbit video of a trained model."""

import argparse

import nerf
import numpy as np
import scenepic as sp
import torch


def _parse_args():
    parser = argparse.ArgumentParser("Orbit Video Maker")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("resolution", type=int, help="Resolution of the video")
    parser.add_argument("mp4_path", help="Path to the output MP4")
    parser.add_argument("--distance", type=int, default=4,
                        help="Distance of the camera")
    parser.add_argument("--fov-y-degrees", type=float, default=40,
                        help="Camera field of view in degrees")
    parser.add_argument("--num-frames", type=int, default=120,
                        help="Number of frames in the video")
    parser.add_argument("--up-dir", default="y+", 
                        choices=["x+", "x-", "y+", "y-", "z+", "z-"],
                        help="The direction that is 'up'")
    parser.add_argument("--alpha-thresh", type=float, default=0.3)
    parser.add_argument("--framerate", type=float, default=15)
    parser.add_argument("--background", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=400*400)
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
    if args.up_dir.startswith("x"):
        right_dir = np.array([0, 1, 0], np.float32)
        forward_dir = np.array([0, 0, -1], np.float32)
    elif args.up_dir.startswith("z"):
        right_dir = np.array([0, 1, 0], np.float32)
        forward_dir = np.array([0, -1, 0], np.float32)
    else:
        right_dir = np.array([1, 0, 0], np.float32)
        forward_dir = np.array([0, 0, -1], np.float32)

    azimuth = np.linspace(0, 4*np.pi, args.num_frames)
    start_value = -np.pi / 12
    mid_value = np.pi / 4
    altitude0 = np.linspace(start_value, mid_value, args.num_frames // 2)
    altitude1 = np.linspace(mid_value, start_value,
                            args.num_frames - args.num_frames // 2)
    altitude = np.concatenate([altitude0, altitude1])

    start_value = 1.2 * args.distance
    mid_value = 0.8 * args.distance
    distances0 = np.linspace(start_value, mid_value, args.num_frames // 2)
    distances1 = np.linspace(mid_value, start_value,
                             args.num_frames - args.num_frames // 2)
    distances = np.concatenate([distances0, distances1])

    fov_y = args.fov_y_degrees * np.pi / 180
    focal_length = .5 * args.resolution / np.tan(.5 * fov_y)

    intrinsics = np.array([
        focal_length, 0, args.resolution / 2,
        0, focal_length, args.resolution / 2,
        0, 0, 1
    ], np.float32).reshape(3, 3)

    bounds_transform = sp.Transforms.scale(2)

    scene = sp.Scene()
    bounds = scene.create_mesh()
    bounds.add_cube(sp.Colors.Blue, transform=bounds_transform)
    canvas = scene.create_canvas_3d(width=800, height=800)
    camera_info = []
    for frame_azi, frame_alt, frame_dist in zip(azimuth, altitude, distances):
        frame = canvas.create_frame()
        position = forward_dir * frame_dist
        elevate = sp.Transforms.rotation_matrix_from_axis_angle(right_dir,
                                                                frame_alt)
        rotate = sp.Transforms.rotation_matrix_from_axis_angle(up_dir,
                                                               frame_azi)
        h_position = np.array([*position, 1], np.float32)
        h_position = rotate @ elevate @ h_position
        position = h_position[:3]

        sp_camera = sp.Camera(position, fov_y_degrees=args.fov_y_degrees)

        extrinsics = sp_camera.camera_to_world @ sp.Transforms.rotation_about_x(np.pi)
        camera = nerf.CameraInfo.create("cam{}".format(len(camera_info)),
                                        (args.resolution, args.resolution),
                                        intrinsics, extrinsics)
        camera_info.append(camera)

        frustum = scene.create_mesh()
        frustum.add_camera_frustum(sp_camera, sp.Colors.White)
        frame.add_mesh(frustum)
        frame.add_mesh(bounds)
        frame.camera = sp_camera

    scene.save_as_html("orbit.html")

    model = nerf.load_model(args.model_path)
    model = model.to("cuda")
    raycaster = nerf.Raycaster(model, isinstance(model, nerf.NeRF))
    sampler = nerf.RaySampler(bounds_transform, camera_info, 128, False, model)
    with sp.VideoWriter(args.mp4_path, camera_info[0].resolution, rgb=True,
                        framerate=args.framerate) as writer:
        with torch.no_grad():
            for frame in range(args.num_frames):
                samples = sampler.rays_for_camera(frame)

                color = []
                alpha = []
                start = 0
                num_rays = len(samples.positions)
                for start in range(0, num_rays, args.batch_size):
                    end = min(start + args.batch_size, num_rays)
                    batch_samples = samples.subset(list(range(start, end)))
                    batch_samples = batch_samples.to("cuda")
                    render = raycaster.render(batch_samples)
                    color.append(render.color.cpu().numpy())
                    alpha.append(render.alpha.cpu().numpy())

                color = np.concatenate(color)
                color = color.reshape(args.resolution, args.resolution, 3)
                color = (color * 255).astype(np.uint8)

                alpha = np.concatenate(alpha)
                alpha = alpha.reshape(args.resolution, args.resolution, 1)
                writer.frame[:] = np.where(alpha > args.alpha_thresh,
                                           color, args.background)
                writer.write_frame()


if __name__ == "__main__":
    _main()
