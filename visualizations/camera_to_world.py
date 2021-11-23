"""Module providing an animation of the camera-to-world transform."""

import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp
from scipy.spatial.transform import Rotation


def camera_to_world(scene: sp.Scene, voxels: ffn.OcTree,
                    camera: ffn.CameraInfo, image: np.ndarray,
                    resolution=400, canvas_id="camera_to_world") -> sp.Canvas3D:
    """Produces an animation of the camera-to-world transform.

    Description:
        The animation will show the camera being rotated and translated
        into world coordinates and then rays being cast from the
        camera origin out into the scene.

    Args:
        scene (sp.Scene): the scene to use for creating scenepic objects
        voxels (ffn.OcTree): the model to use
        camera (ffn.CameraInfo): the camera to display
        image (np.ndarray): the image to display on the camera plane
        resolution (int, optional): the width and height of the canvas.
                                    Defaults to 400.
        canvas_id (str, optional): Id to use for the canvas. Defaults to
                                   "camera_to_world".

    Returns:
        sp.Canvas3D: the canvas object containing the animation
    """
    print("Reading octree")
    canvas = scene.create_canvas_3d(canvas_id,
                                    width=resolution, height=resolution)
    title = scene.create_label(text="Camera to World", size_in_pixels=100,
                               camera_space=True, horizontal_align="center")
    xyz = scene.create_label(text="(X Y Z)", size_in_pixels=50,
                             horizontal_align="center")
    uvw = scene.create_label(text="(u v w)", size_in_pixels=50,
                             horizontal_align="center")

    leaf_centers = voxels.leaf_centers()
    leaf_depths = voxels.leaf_depths()
    leaf_colors = voxels.leaf_data()
    depths = np.unique(leaf_depths)
    model = []
    for depth in depths:
        mesh = scene.create_mesh(layer_id="model")
        transform = sp.Transforms.scale(pow(2., 1-depth) * voxels.scale)
        mesh.add_cube(sp.Colors.White, transform=transform)
        depth_centers = leaf_centers[leaf_depths == depth]
        depth_colors = leaf_colors[leaf_depths == depth]
        mesh.enable_instancing(depth_centers, colors=depth_colors)
        model.append(mesh)

    origin_camera = ffn.CameraInfo("origin",
                                   ffn.Resolution(resolution, resolution),
                                   camera.intrinsics,
                                   np.eye(4, dtype=np.float32))

    sp_image = scene.create_image()
    image = image[..., :3]
    sp_image.from_numpy(image)
    camera_image = scene.create_mesh(texture_id=sp_image.image_id,
                                     double_sided=True)
    camera_image.add_camera_image(origin_camera.to_scenepic())

    frustum = scene.create_mesh()
    frustum.add_camera_frustum(origin_camera.to_scenepic(), sp.Colors.White)

    coord_axes = scene.create_mesh()
    coord_axes.add_coordinate_axes(transform=sp.Transforms.scale(0.5))

    num_frames = 60

    camera_to_world = camera.extrinsics
    qstart = Rotation.from_matrix(np.eye(3)).as_quat()
    qend = Rotation.from_matrix(camera_to_world[:3, :3]).as_quat()
    rot_qs = np.linspace(qstart, qend, num_frames)

    tstart = np.zeros(3, np.float32)
    tend = camera_to_world[:3, 3]
    t_vecs = np.linspace(tstart, tend, num_frames)

    camera_start = [-7, 1, -1]
    lookat = [0, 1, -1]
    fov = 50
    view_cam = sp.Camera(camera_start, lookat, fov_y_degrees=fov)

    def _add_meshes(frame: sp.Frame3D, camera_transform: np.ndarray,
                    camera: sp.Camera):
        for mesh in model:
            frame.add_mesh(mesh)

        frame.add_mesh(frustum, transform=camera_transform)
        frame.add_mesh(camera_image, transform=camera_transform)
        frame.add_mesh(coord_axes)

        pos = np.array([0, -.9, 1, 1], np.float32)
        pos = np.linalg.inv(camera.projection) @ pos
        frame.add_label(title, pos[:3])
        pos = np.array([0, 0.6, 1, 1], np.float32)
        pos = camera_transform @ pos
        frame.add_label(uvw, pos[:3])
        frame.add_label(xyz, [0, -1.2, 0])

    for rot_q in rot_qs:
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = Rotation.from_quat(rot_q).as_matrix()
        frame = canvas.create_frame()
        frame.camera = view_cam
        _add_meshes(frame, transform, frame.camera)

    for t_vec in t_vecs:
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = Rotation.from_quat(rot_qs[-1]).as_matrix()
        transform[:3, 3] = t_vec
        frame = canvas.create_frame()
        frame.camera = view_cam
        _add_meshes(frame, transform, frame.camera)

    print("Intersecting rays")
    vals = np.linspace(0, resolution, 20, endpoint=False).astype(np.int32)
    points = np.stack(np.meshgrid(vals, vals), -1).reshape(-1, 2)
    rays = camera.raycast(points)
    paths = voxels.intersect(rays.origin, rays.direction, 64)
    camera_ray = t_vecs[-1] / np.linalg.norm(t_vecs[-1])
    rotate_axis = np.cross(camera_ray, [-1, 0, 0])

    camera_angles = np.concatenate([
        np.linspace(0, -np.pi / 4, num_frames // 3),
        np.linspace(-np.pi / 4, -np.pi / 4, num_frames - num_frames // 3)
    ])

    ray_meshes = []
    for i in range(num_frames):
        ray_mesh = scene.create_mesh(layer_id="rays")
        for origin, direction, path, t_values in zip(rays.origin,
                                                     rays.direction,
                                                     paths.leaves,
                                                     paths.t_stops):
            first_leaf = (path > -1).argmax()
            if first_leaf == 0:
                continue

            t = (i * t_values[first_leaf]) / (num_frames - 1)
            start = origin
            end = origin + t * direction
            color = leaf_colors[path[first_leaf]]
            ray_mesh.add_thickline(color, start, end, 0.01, 0.01)

        ray_meshes.append(ray_mesh)

    for i in range(num_frames):
        frame = canvas.create_frame()
        frame.add_mesh(ray_meshes[i])
        rotation = sp.Transforms.rotation_matrix_from_axis_angle(rotate_axis,
                                                                 camera_angles[i])
        position = np.ones(4, np.float32)
        position[:3] = camera_start
        position = rotation @ position
        frame.camera = sp.Camera(position[:3], lookat, fov_y_degrees=fov)
        _add_meshes(frame, camera_to_world, frame.camera)

    camera_angles = np.concatenate([
        np.linspace(-np.pi / 4, np.pi / 4, num_frames),
        np.linspace(np.pi / 4, 0, num_frames)
    ])

    camera_fov = np.concatenate([
        np.linspace(fov, 40, num_frames),
        np.linspace(40, fov, num_frames)
    ])

    for i in range(num_frames * 2):
        frame = canvas.create_frame()
        frame.add_mesh(ray_meshes[-1])
        rotation = sp.Transforms.rotation_matrix_from_axis_angle(rotate_axis, camera_angles[i])
        position = np.ones(4, np.float32)
        position[:3] = camera_start
        position = rotation @ position
        frame.camera = sp.Camera(position[:3], lookat, fov_y_degrees=camera_fov[i])
        _add_meshes(frame, camera_to_world, frame.camera)

    return canvas
