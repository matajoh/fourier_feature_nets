"""Component provided logic for triangulating 3D point positions."""

import cv2
import numpy as np
import scenepic as sp
import trimesh


class Triangulation:
    """Class to use for computing triangulated 3D points."""
    def __init__(self, camera_info):
        """Initializer.

        Arguments:
            camera_info: the camera information for the cameras
        """
        self.camera_info = camera_info

    def cast_rays(self, landmarks: np.ndarray, z_near=1, z_far=100) -> np.ndarray:
        """Casts rays into the 3D scene using the landmark positions.

        Arguments:
            landmarks: a (C,L,2) tensor of landmark positions per camera
            z_near: the z-distance of the start of the ray
            z_far: the z-distance of the end of the ray

        Returns:
            a (C,L,2,3) tensor of rays
        """
        rays = []
        for i, info in enumerate(self.camera_info):
            start, end = info.raycast(landmarks[:, i], z_near, z_far)
            rays.append(np.stack([start, end], -2))

        rays = np.stack(rays, axis=1)
        return rays

    def intersect(self, rays: np.ndarray) -> np.ndarray:
        """Finds the nearest point in 3D space to all rays for each landmark.

        Description:
            Finds the point where multiple rays "intersect", defined here as the
            the nearest point in space to all lines. For details see:
            https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#More_than_two_lines
            Variable names are borrowed from those equations for clarity.
        """
        num_frames, num_cameras, num_landmarks = rays.shape[:3]
        seminorm = np.zeros((num_frames, num_cameras, num_landmarks, 3, 3), np.float32)
        seminorm[:] = np.eye(3, dtype=np.float32)
        v = rays[:, :, :, 1] - rays[:, :, :, 0]
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        seminorm -= np.matmul(v.reshape(*rays.shape[:3], 3, 1), v.reshape(*rays.shape[:3], 1, 3))
        seminorm_pinv = np.linalg.pinv(seminorm.sum(1))
        p = rays[:, :, :, 0].reshape(*rays.shape[:3], 3, 1)
        p = np.matmul(seminorm, p).sum(1)
        closest = np.matmul(seminorm_pinv, p).reshape(num_frames, num_landmarks, 3)
        return closest

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """Triangulates 3D landmarks."""
        rays = self.cast_rays(landmarks)
        return self.intersect(rays)

    def save_svt(self, path, camera_info, frames, landmarks, landmarks3d, rays, scan_path=None):
        """Saves a descriptive SVT depicting the accuracy of the triangulation."""
        scene = sp.Scene()
        widths = [info.resolution[0] * 256 // info.resolution[1] for info in camera_info]
        heights = [256] * len(camera_info)
        canvas3ds = [scene.create_canvas_3d(width=width, height=height)
                     for width, height in zip(widths, heights)]
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        frustums = scene.create_mesh()
        svt_cameras = []
        canvas2ds = []
        images = []
        for i, (info, color, width, height) in enumerate(zip(camera_info, colors, widths, heights)):
            cam_landmarks = landmarks[0, i]
            image = scene.create_image()
            pixels = frames[0, i]
            frame_height = pixels.shape[0]
            size = np.array([width * frame_height // height, frame_height], np.float32).reshape(1, 2)
            size = (size - 1) * 0.5
            center = np.array([pixels.shape[1], pixels.shape[0]], np.float32).reshape(1, 2)
            center = (center - 1) * 0.5
            cam_landmarks = (cam_landmarks * size) + center
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            image.from_numpy(pixels)
            images.append(image)

            canvas = scene.create_canvas_2d(width=width, height=height)
            canvas2ds.append(canvas)

            camera = info.to_svt()
            svt_cameras.append(camera)
            frustums.add_camera_frustum(camera, color)

        orbit_canvas = scene.create_canvas_3d(width=512, height=512)

        # show triangulation for each landmark on the first frame
        if scan_path:
            scan = trimesh.load(scan_path)
            scan_mesh = scene.create_mesh(layer_id="scan", shared_color=sp.Colors.Gray)
            verts = scan.triangles.reshape(-1, 3)
            triangles = np.arange(len(verts)).reshape(-1, 3)
            scan_mesh.add_mesh_without_normals(verts, triangles)
        else:
            scan_mesh = None

        estimate_mesh = scene.create_mesh(layer_id="estimates", shared_color=sp.Colors.Yellow)
        estimate_mesh.add_cube(transform=sp.Transforms.scale(0.005))
        estimate_mesh.enable_instancing(landmarks3d[0])

        focus_point = landmarks3d[0].mean(0)

        up_dir = np.array((0, 1, 0), np.float32)
        forward_dir = np.array((0, 0, -1), np.float32)
        orbit_distance = 3 * np.linalg.norm(forward_dir)

        rays = rays[0].swapaxes(1, 0)
        landmarks = landmarks[0].swapaxes(1, 0)
        num_landmarks = len(landmarks)
        for ldmk_id, (cam_rays, cam_landmarks) in enumerate(zip(rays, landmarks)):
            ray_mesh = scene.create_mesh(layer_id="rays")

            orbit_frame = orbit_canvas.create_frame(focus_point=focus_point)
            orbit_frame.add_mesh(frustums)
            orbit_frame.add_mesh(ray_mesh)
            orbit_frame.add_mesh(estimate_mesh)
            if scan_mesh:
                orbit_frame.add_mesh(scan_mesh)

            orbit_rot = sp.Transforms.rotation_matrix_from_axis_angle(up_dir, ldmk_id * np.pi * 2 / num_landmarks)
            orbit_pos = np.dot(orbit_rot[:3, :3], forward_dir)
            orbit_pos = orbit_pos / np.linalg.norm(orbit_pos)
            orbit_pos = focus_point + orbit_pos * orbit_distance
            orbit_frame.camera = sp.Camera(orbit_pos, focus_point, up_dir)

            for camera, ray, canvas3d, color in zip(svt_cameras, cam_rays, canvas3ds, colors):
                frame = canvas3d.create_frame(focus_point=focus_point)
                frame.add_mesh(frustums)
                frame.add_mesh(estimate_mesh)
                frame.add_mesh(ray_mesh)
                if scan_mesh:
                    frame.add_mesh(scan_mesh)

                ray_mesh.add_thickline(color, ray[0], ray[1], 0.02, 0.02)
                frame.camera = camera

            for canvas2d, image, landmark, color, info in zip(canvas2ds, images, cam_landmarks, colors, camera_info):
                center = 0.5 * np.array([256 * info.resolution[0] // info.resolution[1], 256], np.float32)
                frame = canvas2d.create_frame()
                frame.add_image(image)
                landmark = (landmark * center) + center - 0.5
                frame.add_circle(landmark[0], landmark[1], 5, fill_color=color)

        scene.grid("{}px".format(sum(widths) + 4 * len(widths)),
                   "260px 260px 512px",
                   " ".join(["{}px".format(width + 4) for width in widths]))
        scene.place(orbit_canvas.canvas_id, "3", "1 / span 2")
        scene.link_canvas_events(*canvas3ds, *canvas2ds, orbit_canvas)
        scene.save_as_html(path, title="Camcap Triangulation")
