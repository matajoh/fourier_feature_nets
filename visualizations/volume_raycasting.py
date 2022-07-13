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


class VolumeRaycastingAnimation:
    """Creates an animation of a scene being rendered via volumetric raycasting."""
    def __init__(self, dataset: ffn.RayDataset, voxels: ffn.OcTree,
                 num_rays=2000, camera_depth=0.2, num_samples=64,
                 framerate=25, zoom_duration=3, hero_duration=5,
                 casting_duration=8, rendering_duration=6,
                 final_duration=6, rest_duration=2,
                 width=640, height=360, image_resolution=200):
        """Constructor.

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
        self.sampler = dataset.sampler
        self.sp_cameras = [camera.to_scenepic() for camera in dataset.cameras]

        self.scene = sp.Scene()
        self.scene.framerate = framerate

        self.sample_size = 0.01
        self.ray_thickness = 0.001
        self.camera_depth = camera_depth
        self.num_rays = num_rays
        self.aspect_ratio = width / height

        camera_positions = [cam.position for cam in self.sampler.cameras]
        camera_positions = np.concatenate(camera_positions)
        self.hero_cam = self._pick_hero(camera_positions)
        image_height, image_width = dataset.images.shape[1:3]
        resolution = ffn.Resolution(image_width, image_height)
        self.resolution = resolution.scale_to_height(image_resolution)

        self.canvas = self.scene.create_canvas_3d(width=width, height=height)
        self.canvas.shading = sp.Shading(bg_color=sp.Colors.Blue)
        self.canvas.set_layer_settings({
            "camera": {"filled": False},
            "camera_path": {"filled": False}
        })

        self.frames = []
        self.camera_pos = []
        self.camera_lookat = []
        self.camera_fov = []

        self._create_meshes(dataset.images, voxels)
        self._create_rays(dataset, voxels, num_samples)
        self._hero(int(zoom_duration * framerate),
                   int(hero_duration * framerate))
        self._casting(int(casting_duration * framerate))
        self._rendering(int(rendering_duration * framerate))
        self._final(int(final_duration * framerate))
        self._camera_track()
        self._rest(int(rest_duration * framerate))

    def _pick_hero(self, camera_positions: np.ndarray) -> int:
        length = np.linalg.norm(camera_positions, axis=-1, keepdims=True)
        cam_directions = camera_positions / length
        ideal_alt = np.pi / 8
        ideal_direction = [0, np.sin(ideal_alt), np.cos(ideal_alt)]
        distances = np.square(cam_directions - ideal_direction).sum(-1)
        return distances.argmin()

    def _create_meshes(self, images: np.ndarray, voxels: ffn.OcTree):
        self.frustum_meshes = []
        self.image_meshes = []
        for cam, pixels in zip(self.sp_cameras, images):
            frustum = self.scene.create_mesh(layer_id="frustums")
            frustum.add_camera_frustum(cam, sp.Colors.White,
                                       depth=self.camera_depth, thickness=0.001)
            self.frustum_meshes.append(frustum)
            image = self.scene.create_image()

            pixels = cv2.resize(pixels[..., :3], self.resolution)
            image.from_numpy(pixels)
            image_mesh = self.scene.create_mesh(layer_id="images",
                                                texture_id=image.image_id,
                                                double_sided=True)
            image_mesh.add_camera_image(cam, depth=self.camera_depth)
            self.image_meshes.append(image_mesh)

        leaf_centers = voxels.leaf_centers()
        num_parts = 50
        part_index = np.arange(len(leaf_centers)) % num_parts
        np.random.shuffle(part_index)

        self.model_meshes = []
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

                mesh = self.scene.create_mesh(layer_id="model")
                transform = sp.Transforms.scale(pow(2., 1-depth) * voxels.scale)
                mesh.add_cube(sp.Colors.White, transform=transform)
                depth_centers = leaf_centers[cam_leaves & depth_leaves]
                depth_colors = leaf_colors[cam_leaves & depth_leaves]
                mesh.enable_instancing(depth_centers, colors=depth_colors)
                cam_meshes.append(mesh)

            self.model_meshes.append(cam_meshes)

    def _create_rays(self, dataset: ffn.RayDataset, voxels: ffn.OcTree,
                     num_samples: int):
        hero = self.hero_cam * self.sampler.rays_per_camera
        hero += self.sampler.rays_per_camera // 2
        hero += dataset.resolution.width // 2
        not_empty = np.nonzero(dataset.alphas > 0)
        ray_index = np.linspace(0, len(not_empty), self.num_rays - 1,
                                endpoint=False).astype(np.int32)
        ray_index = not_empty[ray_index].reshape(-1)
        ray_index = np.concatenate([[hero], ray_index])
        self.actual_colors = dataset.colors[ray_index].unsqueeze(1).numpy()
        starts = self.sampler.starts[ray_index].numpy()
        directions = self.sampler.directions[ray_index].numpy()

        print("Performing ray intersection")
        path = voxels.intersect(starts, directions, num_samples - 1)
        t_values = path.t_stops
        t_values = np.concatenate([np.full_like(t_values[:, :1],
                                                self.camera_depth),
                                   t_values], -1)
        self.starts = starts.reshape(self.num_rays, 1, 3)
        self.directions = directions.reshape(self.num_rays, 1, 3)
        self.t_values = t_values.reshape(self.num_rays, num_samples, 1)
        self.positions = self.starts + self.directions * self.t_values
        leaf_ids = path.leaves.reshape(-1)
        leaf_colors = voxels.leaf_data()
        colors = [[0.1, 0.1, 0.1] if i < 0 else leaf_colors[i]
                  for i in leaf_ids]
        colors = np.array(colors, np.float32)
        colors = colors.reshape(self.num_rays, num_samples - 1, 3)
        self.colors = np.concatenate([np.zeros_like(colors[:, :1]), colors], 1)

        mid_t_values = np.linspace(self.camera_depth,
                                   self.camera_depth + num_samples * self.sample_size,
                                   num_samples)
        self.mid_t_values = mid_t_values.reshape(1, num_samples, 1)
        self.mid_positions = self.starts + self.mid_t_values * self.directions
        self.near = self.camera_depth
        self.far = self.t_values.max()
        self.hero_forward = self.directions[0, 0]
        self.hero_right = np.cross(self.hero_forward, [0, 1, 0])

    def _hero(self, num_zoom_frames: int, num_hero_frames: int):
        first_sample = self.t_values[0, 1]
        frame_t_values = np.concatenate([
            np.linspace(self.near, first_sample, num_zoom_frames,
                        endpoint=False),
            np.linspace(first_sample, self.far, num_hero_frames)
        ])

        hero_pos = (self.starts[0] +
                    frame_t_values.reshape(-1, 1) * self.directions[0])

        camera_start = (-0.5 * self.hero_forward +
                        0.2 * self.hero_right + hero_pos[0])
        camera_end = (-0.4 * self.hero_forward +
                      0.025 * self.hero_right + hero_pos[-1])
        self.camera_pos.append((0, camera_start))
        self.camera_pos.append((len(frame_t_values), camera_end))

        self.camera_lookat.append((0, hero_pos[0]))
        for pos in hero_pos:
            self.camera_lookat.append((1, pos))

        self.camera_fov.append((0, 75))
        self.camera_fov.append((num_zoom_frames, 40))
        self.camera_fov.append((num_hero_frames, 40))

        start = self.starts.reshape(self.num_rays, 3)[0]
        direction = self.directions.reshape(self.num_rays, 3)[0]

        bar = ffn.ETABar("Hero Sequence", max=num_hero_frames + num_zoom_frames)
        for _, frame_t in enumerate(frame_t_values):
            bar.next()
            frame = self.canvas.create_frame()
            self.frames.append(frame)
            for frustum, image in zip(self.frustum_meshes, self.image_meshes):
                frame.add_mesh(frustum)
                frame.add_mesh(image)

            ray_mesh = self.scene.create_mesh(layer_id="rays")
            ray_mesh.add_thickline(sp.Colors.White,
                                   start + self.camera_depth * direction,
                                   start + frame_t * direction,
                                   self.ray_thickness, self.ray_thickness)

            frame.add_mesh(ray_mesh)

            frame_pos = np.where(self.t_values < frame_t, self.positions, 0)
            hero_pos = frame_pos[0].reshape(-1, 3)
            hero_color = self.colors[0].reshape(-1, 3)

            valid = (hero_pos != 0).any(-1)
            hero_pos = hero_pos[valid]
            hero_color = hero_color[valid]

            if len(frame_pos):
                sample_mesh = self.scene.create_mesh(layer_id="samples")
                sample_mesh.add_sphere(sp.Colors.Black,
                                       transform=sp.Transforms.scale(self.sample_size * 2))
                sample_mesh.enable_instancing(hero_pos, colors=hero_color)
                frame.add_mesh(sample_mesh)

        bar.finish()

    def _casting(self, num_casting_frames: int):
        # zoom out to wide shot
        num_pan_frames = num_casting_frames // 2
        num_zoom_frames = num_casting_frames - num_pan_frames
        camera_mid = -1 * self.hero_forward + 0.5 * self.hero_right
        camera_end = -2 * self.hero_forward + 1 * self.hero_right
        self.camera_pos.append((num_pan_frames, camera_mid))
        self.camera_pos.append((num_zoom_frames, camera_end))

        self.camera_lookat.append((num_casting_frames, [0, 0, 0]))

        self.camera_fov.append((num_pan_frames, 40))
        self.camera_fov.append((num_zoom_frames, 75))

        bar = ffn.ETABar("Casting", max=num_casting_frames)
        frame_t_values = np.linspace(self.near, self.far, num_casting_frames)
        for i, frame_t in enumerate(frame_t_values):
            bar.next()
            frame = self.canvas.create_frame()
            self.frames.append(frame)

            for frustum, image in zip(self.frustum_meshes, self.image_meshes):
                frame.add_mesh(frustum)
                frame.add_mesh(image)

            ray_start = self.starts + frame_t_values[max(0, i - 4)] * self.directions
            ray_end = self.starts + frame_t * self.directions

            ray_mesh = self.scene.create_mesh(layer_id="rays")
            ray_mesh.add_thickline(sp.Colors.White, self.starts[0],
                                   self.starts[0] + self.far * self.directions[0],
                                   self.ray_thickness, self.ray_thickness)

            for r in range(1, self.num_rays):
                ray_mesh.add_thickline(sp.Colors.Gray, ray_start[r], ray_end[r],
                                       self.ray_thickness, self.ray_thickness)

            frame.add_mesh(ray_mesh)

            frame_pos = np.where(self.t_values < frame_t, self.positions, 0)
            hero_pos = self.positions[0].reshape(-1, 3)
            hero_colors = self.colors[0].reshape(-1, 3)

            frame_pos = frame_pos[1:]
            frame_color = self.colors[1:]

            valid = (frame_color != 0.1).any(-1)
            frame_pos = np.concatenate([hero_pos, frame_pos[valid]])
            frame_color = np.concatenate([hero_colors, frame_color[valid]])

            size = _lerp(min(i, num_pan_frames), num_pan_frames,
                         2 * self.sample_size, self.sample_size)

            valid = (frame_pos != 0).any(-1)
            frame_pos = frame_pos[valid]
            frame_color = frame_color[valid]
            if len(frame_pos):
                sample_mesh = self.scene.create_mesh(layer_id="samples")
                sample_mesh.add_sphere(sp.Colors.Black,
                                       transform=sp.Transforms.scale(size))
                sample_mesh.enable_instancing(frame_pos, colors=frame_color)
                frame.add_mesh(sample_mesh)

        bar.finish()

    def _rendering(self, num_rendering_frames: int):
        num_zoom_frames = num_rendering_frames // 4
        num_follow_frames = num_rendering_frames - num_zoom_frames
        _, camera_mid = self.camera_pos[-1]
        camera_end = 0.4 * self.hero_forward + 0.025 * self.hero_right + self.mid_positions[0, -1]
        self.camera_pos.append((num_zoom_frames, camera_mid))
        self.camera_pos.append((num_follow_frames, camera_end))

        self.camera_lookat.append((num_rendering_frames, self.mid_positions[0, -1]))

        self.camera_fov.append((num_zoom_frames, 75))
        self.camera_fov.append((num_follow_frames, 40))

        bar = ffn.ETABar("Rendering", max=num_rendering_frames)
        for i in range(num_rendering_frames):
            bar.next()
            frame = self.canvas.create_frame()
            self.frames.append(frame)

            for frustum, image in zip(self.frustum_meshes, self.image_meshes):
                frame.add_mesh(frustum)
                frame.add_mesh(image)

            sample_mesh = self.scene.create_mesh(layer_id="samples")
            sample_mesh.add_sphere(sp.Colors.Black, transform=sp.Transforms.scale(self.sample_size))
            frame_pos = _lerp(i, num_rendering_frames, self.positions, self.mid_positions)
            ray_mesh = self.scene.create_mesh(layer_id="rays")
            start = frame_pos[0, 0]
            end = frame_pos[0, -1]
            ray_mesh.add_thickline(sp.Colors.White, start, end,
                                   self.ray_thickness, self.ray_thickness)
            frame.add_mesh(ray_mesh)

            hero_pos = frame_pos[0].reshape(-1, 3)
            hero_colors = self.colors[0].reshape(-1, 3)
            frame_pos = frame_pos[1:].reshape(-1, 3)
            frame_colors = self.colors[1:].reshape(-1, 3)
            valid = (frame_colors != 0.1).any(-1)
            frame_pos = np.concatenate([hero_pos, frame_pos[valid]])
            frame_colors = np.concatenate([hero_colors, frame_colors[valid]])
            sample_mesh.enable_instancing(frame_pos, colors=frame_colors)
            frame.add_mesh(sample_mesh)

        bar.finish()

    def _final(self, num_final_frames: int):
        lookat = self.positions[0, 0]
        num_watch_frames = num_final_frames // 4
        num_return_frames = num_final_frames - num_watch_frames
        camera_mid = 0.3 * self.hero_forward + lookat
        camera_end = -0.5 * self.hero_forward + 0.2 * self.hero_right + lookat
        self.camera_pos.append((num_watch_frames, camera_mid))
        self.camera_pos.append((num_return_frames, camera_end))

        self.camera_lookat.append((num_watch_frames, lookat))
        self.camera_lookat.append((num_return_frames, lookat))

        self.camera_fov.append((num_watch_frames, 40))
        self.camera_fov.append((num_return_frames, 75))

        final_positions = self.starts + self.camera_depth * self.directions
        final_positions = final_positions.reshape(self.num_rays, 1, 3)
        model_start = num_final_frames - len(self.model_meshes)
        bar = ffn.ETABar("Final", max=num_final_frames)
        for i in range(num_final_frames):
            bar.next()
            frame = self.canvas.create_frame()
            self.frames.append(frame)

            for frustum, image in zip(self.frustum_meshes, self.image_meshes):
                frame.add_mesh(frustum)
                frame.add_mesh(image)

            if i < num_watch_frames:
                frame_pos = _lerp(min(i, num_watch_frames), num_watch_frames, self.mid_positions, final_positions)
                lerp_colors = _lerp(min(i, num_watch_frames), num_watch_frames, self.colors, self.actual_colors)
                ray_mesh = self.scene.create_mesh(layer_id="rays")
                start = frame_pos[0, 0]
                end = frame_pos[0, -1]
                ray_mesh.add_thickline(sp.Colors.White, start, end,
                                       self.ray_thickness, self.ray_thickness)
                frame.add_mesh(ray_mesh)

                sample_mesh = self.scene.create_mesh(layer_id="samples")
                size = _lerp(i, num_watch_frames, self.sample_size, 0.0001)
                sample_mesh.add_sphere(sp.Colors.Black, transform=sp.Transforms.scale(size))

                hero_pos = frame_pos[0].reshape(-1, 3)
                hero_colors = lerp_colors[0].reshape(-1, 3)
                frame_pos = frame_pos[1:].reshape(-1, 3)
                frame_colors = self.colors[1:].reshape(-1, 3)
                lerp_colors = lerp_colors[1:].reshape(-1, 3)
                valid = (frame_colors != 0.1).any(-1)
                frame_pos = np.concatenate([hero_pos, frame_pos[valid]])
                frame_colors = np.concatenate([hero_colors, lerp_colors[valid]])
                sample_mesh.enable_instancing(frame_pos, colors=frame_colors)
                frame.add_mesh(sample_mesh)

            if i > model_start:
                for j in range(0, i - model_start):
                    for mesh in self.model_meshes[j]:
                        frame.add_mesh(mesh)

        bar.finish()

    def _camera_track(self):
        self.camera_pos = _interp(self.camera_pos)
        self.camera_fov = _interp(self.camera_fov)
        self.camera_lookat = _interp(self.camera_lookat)
        path_mesh = self.scene.create_mesh(layer_id="camera_path")
        last_pos = self.camera_pos[0]
        last_lookat = self.camera_lookat[0]
        bar = ffn.ETABar("Camera Track")
        for frame, pos, fov, lookat in zip(self.frames, self.camera_pos, self.camera_fov, self.camera_lookat):
            bar.next()
            camera = sp.Camera(pos, lookat, aspect_ratio=self.aspect_ratio,
                               fov_y_degrees=fov)
            frame.camera = camera

            path_mesh.add_thickline(sp.Colors.Red, last_pos, pos, 0.005, 0.005)
            last_pos = pos

            path_mesh.add_thickline(sp.Colors.Green, last_lookat, lookat, 0.005, 0.005)
            last_lookat = lookat

            cam_mesh = self.scene.create_mesh(layer_id="camera")
            cam_mesh.add_camera_frustum(camera, sp.Colors.Red)
            frame.add_mesh(cam_mesh)
            frame.add_mesh(path_mesh)

        bar.finish()

    def _rest(self, num_rest_frames: int):
        for _ in range(num_rest_frames):
            frame = self.canvas.create_frame()

            for cam_meshes in self.model_meshes:
                for mesh in cam_meshes:
                    frame.add_mesh(mesh)

            for frustum, image in zip(self.frustum_meshes, self.image_meshes):
                frame.add_mesh(frustum)
                frame.add_mesh(image)

            camera = sp.Camera(self.camera_pos[-1],
                               look_at=self.camera_lookat[-1],
                               aspect_ratio=self.aspect_ratio,
                               fov_y_degrees=self.camera_fov[-1])
            frame.camera = camera

            cam_mesh = self.scene.create_mesh(layer_id="camera")
            cam_mesh.add_camera_frustum(camera, sp.Colors.Red)
            frame.add_mesh(cam_mesh)

    def save_as_html(self, path):
        """Save the animation as an HTML page."""
        self.scene.save_as_html(path, "Volume Raycasting")


if __name__ == "__main__":
    dataset = ffn.ImageDataset.load("antinous_800.npz", "train", 64, True, False)
    voxels = ffn.OcTree.load("antinous_octree_8.npz")
    anim = VolumeRaycastingAnimation(dataset, voxels, width=1280, height=720)
    print("Writing scenepic to file...")
    anim.save_as_html("volume_raycasting.html")
