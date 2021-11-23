"""Animation of the volume raycasting process."""

import cv2
import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp


def _lerp(i, end, values0, values1):
    beta = i / (end - 1)
    alpha = 1 - beta
    return alpha * values0 + beta * values1


def _interp(stops):
    x = np.array([stop[0] for stop in stops])
    y = np.stack([stop[1] for stop in stops])

    values = []
    for steps, current_val, next_val in zip(x[1:], y[:-1], y[1:]):
        values.append(np.linspace(current_val, next_val, steps, endpoint=False))

    values = np.concatenate(values)
    for _ in range(5):
        new_values = np.zeros_like(values)
        new_values[0] = (values[:3].sum(0) + 2*values[0]) / 5
        new_values[1] = (values[:4].sum(0) + values[0]) / 5
        new_values[-1] = (values[-3:].sum(0) + 2 * values[-1]) / 5
        new_values[-2] = (values[-4:].sum(0) + values[-1]) / 5
        for i in range(2, len(values) - 2):
            new_values[i] = values[i-2:i+3].mean(0)

        values = new_values

    return values


def volume_raycasting(dataset: ffn.RayDataset, voxels: ffn.OcTree,
                      num_rays=2000, camera_depth=0.2, num_samples=64,
                      framerate=25, zoom_duration=3, hero_duration=5,
                      casting_duration=8, rendering_duration=6,
                      final_duration=6, rest_duration=2,
                      width=640, height=360, image_resolution=200) -> sp.Scene:
    """Creates an animation of a scene being rendered via volumetric raycasting.

    Args:
        dataset (ffn.RayDataset): The dataset to use for sourcing images and
                                   camera positions.
        voxels (ffn.OcTree): An octree modeling the object corresponding to
                              the dataset.
        num_rays (int, optional): Number of rays to cast and animate.
                                  Defaults to 2000.
        camera_depth (float, optional): Distance to use for frustums and camera
                                        image placement. Defaults to 0.2.
        num_samples (int, optional): The number of samples to animate per ray.
                                     Defaults to 64.
        framerate (float, optional): Target framerate for the scene.
                                   Defaults to 25.
        zoom_duration (float, optional): Duration of the initial zoom animation
                                       in seconds. Defaults to 3.
        hero_duration (float, optional): Number of seconds to follow the "hero"
                                         ray. Defaults to 5.
        casting_duration (float, optional): Number of seconds to animate the
                                            casting process. Defaults to 8.
        rendering_duration (float, optional): Number of seconds to animate the
                                              rendering process. Defaults to 6.
        final_duration (float, optional): Number of seconds to render the final
                                          camera movement. Defaults to 6.
        rest_duration (float, optional): Time at the end to rest. Defaults to 2.
        width (int, optional): Width of the scenepic canvas. Defaults to 640.
        height (int, optional): Height of the scenepic canvas. Defaults to 360.
        image_resolution (int, optional): Resolution of the camera images.
                                          Defaults to 200.

    Returns:
        sp.Scene: returns the constructed scenepic scene
    """
    # TODO this is messy and needs cleanup.

    sampler = dataset.sampler
    sp_cameras = [camera.to_scenepic() for camera in dataset.cameras]

    scene = sp.Scene()
    scene.framerate = framerate

    num_zoom_frames = int(zoom_duration * framerate)
    num_hero_frames = int(hero_duration * framerate)
    num_casting_frames = int(casting_duration * framerate)
    num_rendering_frames = int(rendering_duration * framerate)
    num_final_frames = int(final_duration * framerate)
    num_rest_frames = int(rest_duration * framerate)

    camera_positions = np.concatenate([cam.position for cam in sampler.cameras])
    hero_cam = camera_positions[:, 2].argmax()
    height, width = dataset.images.shape[1:3]
    resolution = ffn.Resolution(width, height)
    resolution = resolution.scale_to_height(image_resolution)

    frustum_meshes = []
    image_meshes = []
    for cam, pixels in zip(sp_cameras, dataset.images):
        frustum = scene.create_mesh(layer_id="frustums")
        frustum.add_camera_frustum(cam, sp.Colors.White,
                                   depth=camera_depth, thickness=0.001)
        frustum_meshes.append(frustum)
        image = scene.create_image()

        pixels = cv2.resize(pixels[..., :3], resolution)
        image.from_numpy(pixels)
        image_mesh = scene.create_mesh(layer_id="images",
                                       texture_id=image.image_id,
                                       double_sided=True)
        image_mesh.add_camera_image(cam, depth=camera_depth)
        image_meshes.append(image_mesh)

    leaf_centers = voxels.leaf_centers()
    num_parts = 50
    part_index = np.arange(len(leaf_centers)) % num_parts
    np.random.shuffle(part_index)

    model_meshes = []
    leaf_depths = voxels.leaf_depths()
    leaf_colors = voxels.leaf_data()
    depths = np.unique(leaf_depths)
    for i in range(num_parts):
        cam_meshes = []
        cam_leaves = part_index == i
        if not cam_leaves.any():
            continue

        for depth in depths:
            depth_leaves = leaf_depths == depth
            if not (cam_leaves & depth_leaves).any():
                continue

            mesh = scene.create_mesh(layer_id="model")
            transform = sp.Transforms.scale(pow(2., 1-depth) * voxels.scale)
            mesh.add_cube(sp.Colors.White, transform=transform)
            depth_centers = leaf_centers[cam_leaves & depth_leaves]
            depth_colors = leaf_colors[cam_leaves & depth_leaves]
            mesh.enable_instancing(depth_centers, colors=depth_colors)
            cam_meshes.append(mesh)

        model_meshes.append(cam_meshes)

    hero = hero_cam * sampler.rays_per_camera + sampler.rays_per_camera // 2
    hero += dataset.resolution.width // 2
    not_empty = np.nonzero(dataset.alphas > 0)
    ray_index = np.linspace(0, len(not_empty), num_rays - 1,
                            endpoint=False).astype(np.int32)
    ray_index = not_empty[ray_index].reshape(-1)
    ray_index = np.concatenate([[hero], ray_index])
    actual_colors = dataset.colors[ray_index].unsqueeze(1).numpy()
    starts = sampler.starts[ray_index].numpy()
    directions = sampler.directions[ray_index].numpy()

    print("Performing ray intersection")
    path = voxels.intersect(starts, directions, num_samples - 1)
    t_values = path.t_stops
    first_sample = t_values[0].min()
    t_values = np.concatenate([np.full_like(t_values[:, :1], camera_depth),
                               t_values], -1)
    starts = starts.reshape(num_rays, 1, 3)
    directions = directions.reshape(num_rays, 1, 3)
    t_values = t_values.reshape(num_rays, num_samples, 1)
    positions = starts + directions * t_values
    leaf_ids = path.leaves.reshape(-1)
    colors = [[0.1, 0.1, 0.1] if i < 0 else leaf_colors[i]
              for i in leaf_ids]
    colors = np.array(colors, np.float32)
    colors = colors.reshape(num_rays, num_samples - 1, 3)
    colors = np.concatenate([np.zeros_like(colors[:, :1]), colors], 1)

    canvas = scene.create_canvas_3d(width=width, height=height)
    canvas.shading = sp.Shading(bg_color=sp.Colors.Blue)

    sample_size = 0.01
    ray_thickness = 0.001
    mid_t_values = np.linspace(camera_depth,
                               camera_depth + num_samples * sample_size,
                               num_samples)
    mid_t_values = mid_t_values.reshape(1, num_samples, 1)
    mid_positions = starts + mid_t_values * directions
    near = camera_depth
    far = t_values.max()
    frame_t_values = np.concatenate([
        np.linspace(near, first_sample, num_zoom_frames, endpoint=False),
        np.linspace(first_sample, far, num_hero_frames, endpoint=False)
    ])

    hero_pos = (starts[0] + frame_t_values.reshape(-1, 1) * directions[0])
    hero_forward = directions[0, 0]
    hero_right = np.cross(hero_forward, [0, 1, 0])

    frames = []
    camera_pos = []
    camera_lookat = []
    camera_fov = []

    camera_start = -0.5 * hero_forward + 0.2 * hero_right + hero_pos[0]
    camera_end = -0.4 * hero_forward + 0.025 * hero_right + hero_pos[-1]
    camera_pos.append((0, camera_start))
    camera_pos.append((len(frame_t_values), camera_end))

    camera_lookat.append((0, hero_pos[0]))
    for pos in hero_pos:
        camera_lookat.append((1, pos))

    camera_fov.append((0, 75))
    camera_fov.append((num_zoom_frames, 40))
    camera_fov.append((num_hero_frames, 40))

    starts = starts.reshape(num_rays, 3)
    directions = directions.reshape(num_rays, 3)

    bar = ffn.ETABar("Hero Sequence", max=num_hero_frames + num_zoom_frames)
    for i, frame_t in enumerate(frame_t_values):
        bar.next()
        frame = canvas.create_frame()
        frames.append(frame)
        for frustum, image in zip(frustum_meshes, image_meshes):
            frame.add_mesh(frustum)
            frame.add_mesh(image)

        ray_start = starts + frame_t_values[max(0, i - 4)] * directions
        ray_end = starts + frame_t * directions

        ray_mesh = scene.create_mesh(layer_id="rays")
        ray_mesh.add_thickline(sp.Colors.White,
                               starts[0] + camera_depth * directions[0],
                               ray_end[0], ray_thickness, ray_thickness)

        frame.add_mesh(ray_mesh)

        frame_pos = np.where(t_values < frame_t, positions, 0)
        hero_pos = frame_pos[0].reshape(-1, 3)
        hero_color = colors[0].reshape(-1, 3)

        valid = (hero_pos != 0).any(-1)
        hero_pos = hero_pos[valid]
        hero_color = hero_color[valid]

        if len(frame_pos):
            sample_mesh = scene.create_mesh(layer_id="samples")
            sample_mesh.add_sphere(sp.Colors.Black, transform=sp.Transforms.scale(sample_size * 2))
            sample_mesh.enable_instancing(hero_pos, colors=hero_color)
            frame.add_mesh(sample_mesh)

    bar.finish()

    # zoom out to wide shot
    num_pan_frames = num_casting_frames // 2
    num_zoom_frames = num_casting_frames - num_pan_frames
    camera_start = camera_pos[-1]
    camera_mid = -1 * hero_forward + 0.5 * hero_right
    camera_end = -2 * hero_forward + 1 * hero_right
    camera_pos.append((num_pan_frames, camera_mid))
    camera_pos.append((num_zoom_frames, camera_end))

    camera_lookat.append((num_casting_frames, [0, 0, 0]))

    camera_fov.append((num_pan_frames, 40))
    camera_fov.append((num_zoom_frames, 75))

    bar = ffn.ETABar("Casting", max=num_casting_frames)
    frame_t_values = np.linspace(near, far, num_casting_frames)
    for i, frame_t in enumerate(frame_t_values):
        bar.next()
        frame = canvas.create_frame()
        frames.append(frame)

        for frustum, image in zip(frustum_meshes, image_meshes):
            frame.add_mesh(frustum)
            frame.add_mesh(image)

        ray_start = starts + frame_t_values[max(0, i - 4)] * directions
        ray_end = starts + frame_t * directions

        ray_mesh = scene.create_mesh(layer_id="rays")
        ray_mesh.add_thickline(sp.Colors.White, starts[0],
                               starts[0] + far * directions[0],
                               ray_thickness, ray_thickness)

        for r in range(1, num_rays):
            ray_mesh.add_thickline(sp.Colors.Gray, ray_start[r], ray_end[r],
                                   ray_thickness, ray_thickness)

        frame.add_mesh(ray_mesh)

        frame_pos = np.where(t_values < frame_t, positions, 0)
        hero_pos = positions[0].reshape(-1, 3)
        hero_colors = colors[0].reshape(-1, 3)

        frame_pos = frame_pos[1:]
        frame_color = colors[1:]

        valid = (frame_color != 0.1).any(-1)
        frame_pos = np.concatenate([hero_pos, frame_pos[valid]])
        frame_color = np.concatenate([hero_colors, frame_color[valid]])

        size = _lerp(min(i, num_pan_frames), num_pan_frames, 2 * sample_size, sample_size)

        valid = (frame_pos != 0).any(-1)
        frame_pos = frame_pos[valid]
        frame_color = frame_color[valid]
        if len(frame_pos):
            sample_mesh = scene.create_mesh(layer_id="samples")
            sample_mesh.add_sphere(sp.Colors.Black, transform=sp.Transforms.scale(size))
            sample_mesh.enable_instancing(frame_pos, colors=frame_color)
            frame.add_mesh(sample_mesh)

    bar.finish()

    num_zoom_frames = num_rendering_frames // 4
    num_follow_frames = num_rendering_frames - num_zoom_frames
    camera_start = camera_end
    camera_mid = camera_start
    camera_end = 0.4 * hero_forward + 0.025 * hero_right + mid_positions[0, -1]
    camera_pos.append((num_zoom_frames, camera_mid))
    camera_pos.append((num_follow_frames, camera_end))

    camera_lookat.append((num_rendering_frames, mid_positions[0, -1]))

    camera_fov.append((num_zoom_frames, 75))
    camera_fov.append((num_follow_frames, 40))

    bar = ffn.ETABar("Rendering", max=num_rendering_frames)
    for i in range(num_rendering_frames):
        bar.next()
        frame = canvas.create_frame()
        frames.append(frame)

        for frustum, image in zip(frustum_meshes, image_meshes):
            frame.add_mesh(frustum)
            frame.add_mesh(image)

        sample_mesh = scene.create_mesh(layer_id="samples")
        sample_mesh.add_sphere(sp.Colors.Black, transform=sp.Transforms.scale(sample_size))
        frame_pos = _lerp(i, num_rendering_frames, positions, mid_positions)
        ray_mesh = scene.create_mesh(layer_id="rays")
        start = frame_pos[0, 0]
        end = frame_pos[0, -1]
        ray_mesh.add_thickline(sp.Colors.White, start, end,
                               ray_thickness, ray_thickness)
        frame.add_mesh(ray_mesh)

        hero_pos = frame_pos[0].reshape(-1, 3)
        hero_colors = colors[0].reshape(-1, 3)
        frame_pos = frame_pos[1:].reshape(-1, 3)
        frame_colors = colors[1:].reshape(-1, 3)
        valid = (frame_colors != 0.1).any(-1)
        frame_pos = np.concatenate([hero_pos, frame_pos[valid]])
        frame_colors = np.concatenate([hero_colors, frame_colors[valid]])
        sample_mesh.enable_instancing(frame_pos, colors=frame_colors)
        frame.add_mesh(sample_mesh)

    bar.finish()

    num_watch_frames = num_final_frames // 4
    num_return_frames = num_final_frames - num_watch_frames
    camera_start = camera_end
    camera_mid = 0.3 * hero_forward + hero_pos[0]
    camera_end = -0.5 * hero_forward + 0.2 * hero_right + hero_pos[0]
    camera_pos.append((num_watch_frames, camera_mid))
    camera_pos.append((num_return_frames, camera_end))

    camera_lookat.append((num_watch_frames, hero_pos[0]))
    camera_lookat.append((num_return_frames, hero_pos[0]))

    camera_fov.append((num_watch_frames, 40))
    camera_fov.append((num_return_frames, 75))

    final_positions = starts + camera_depth * directions
    final_positions = final_positions.reshape(num_rays, 1, 3)
    model_start = num_final_frames - len(model_meshes)
    for i in range(num_final_frames):
        frame = canvas.create_frame()
        frames.append(frame)

        for frustum, image in zip(frustum_meshes, image_meshes):
            frame.add_mesh(frustum)
            frame.add_mesh(image)

        if i < num_watch_frames:
            frame_pos = _lerp(min(i, num_watch_frames), num_watch_frames, mid_positions, final_positions)
            lerp_colors = _lerp(min(i, num_watch_frames), num_watch_frames, colors, actual_colors)
            ray_mesh = scene.create_mesh(layer_id="rays")
            start = frame_pos[0, 0]
            end = frame_pos[0, -1]
            ray_mesh.add_thickline(sp.Colors.White, start, end,
                                   ray_thickness, ray_thickness)
            frame.add_mesh(ray_mesh)

            sample_mesh = scene.create_mesh(layer_id="samples")
            size = _lerp(i, num_watch_frames, sample_size, 0.0001)
            sample_mesh.add_sphere(sp.Colors.Black, transform=sp.Transforms.scale(size))

            hero_pos = frame_pos[0].reshape(-1, 3)
            hero_colors = lerp_colors[0].reshape(-1, 3)
            frame_pos = frame_pos[1:].reshape(-1, 3)
            frame_colors = colors[1:].reshape(-1, 3)
            lerp_colors = lerp_colors[1:].reshape(-1, 3)
            valid = (frame_colors != 0.1).any(-1)
            frame_pos = np.concatenate([hero_pos, frame_pos[valid]])
            frame_colors = np.concatenate([hero_colors, lerp_colors[valid]])
            sample_mesh.enable_instancing(frame_pos, colors=frame_colors)
            frame.add_mesh(sample_mesh)

        if i > model_start:
            for j in range(0, i - model_start):
                for mesh in model_meshes[j]:
                    frame.add_mesh(mesh)

    camera_pos = _interp(camera_pos)
    camera_fov = _interp(camera_fov)
    camera_lookat = _interp(camera_lookat)
    path_mesh = scene.create_mesh(layer_id="camera_path")
    last_pos = camera_pos[0]
    last_lookat = camera_lookat[0]
    for frame, pos, fov, lookat in zip(frames, camera_pos, camera_fov, camera_lookat):
        camera = sp.Camera(pos, lookat, aspect_ratio=width/height,
                           fov_y_degrees=fov)
        frame.camera = camera

        path_mesh.add_thickline(sp.Colors.Red, last_pos, pos, 0.005, 0.005)
        last_pos = pos

        path_mesh.add_thickline(sp.Colors.Green, last_lookat, lookat, 0.005, 0.005)
        last_lookat = lookat

        cam_mesh = scene.create_mesh(layer_id="camera")
        cam_mesh.add_camera_frustum(camera, sp.Colors.Red)
        frame.add_mesh(cam_mesh)
        frame.add_mesh(path_mesh)

    for _ in range(num_rest_frames):
        frame = canvas.create_frame()

        for cam_meshes in model_meshes:
            for mesh in cam_meshes:
                frame.add_mesh(mesh)

        for frustum, image in zip(frustum_meshes, image_meshes):
            frame.add_mesh(frustum)
            frame.add_mesh(image)

        camera = sp.Camera(camera_pos[-1], look_at=camera_lookat[-1],
                           aspect_ratio=width/height,
                           fov_y_degrees=camera_fov[-1])
        frame.camera = camera

        cam_mesh = scene.create_mesh(layer_id="camera")
        cam_mesh.add_camera_frustum(camera, sp.Colors.Red)
        frame.add_mesh(cam_mesh)

    canvas.set_layer_settings({
        "camera": {"filled": False},
        "camera_path": {"filled": False}
    })

    return scene
