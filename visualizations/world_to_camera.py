"""Animation of the world-to-camera transform."""

import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp
from scipy.spatial.transform import Rotation


def world_to_camera(scene: sp.Scene, voxels: ffn.OcTree,
                    camera: ffn.CameraInfo, image: np.ndarray,
                    resolution=400, canvas_id="world_to_camera") -> sp.Canvas3D:
    """Creates an animation of the world-to-camera transform.

    Args:
        scene (sp.Scene): The object to use for creating scenepic objects.
        voxels (ffn.OcTree): An octree describing a model to transform
        camera (ffn.CameraInfo): The camera to use
        image (np.ndarray): The image to render on the camera plane
        resolution (int, optional): The canvas resolution. Defaults to 400.
        canvas_id (str, optional): ID for the canvas.
                                   Defaults to "world_to_camera".

    Returns:
        sp.Canvas3D: a canvas containing the animation.
    """
    canvas = scene.create_canvas_3d(canvas_id,
                                    width=resolution, height=resolution)
    title = scene.create_label(text="World to Camera", size_in_pixels=100,
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

    sp_image = scene.create_image()
    image = image[..., :3]
    sp_image.from_numpy(image)
    camera_image = scene.create_mesh(texture_id=sp_image.image_id,
                                     double_sided=True)
    camera_image.add_camera_image(camera.to_scenepic())

    frustum = scene.create_mesh()
    frustum.add_camera_frustum(camera.to_scenepic(), sp.Colors.White)

    coord_axes = scene.create_mesh()
    coord_axes.add_coordinate_axes(transform=sp.Transforms.scale(0.5))

    num_frames = 60

    world_to_camera = np.linalg.inv(camera.extrinsics)
    qstart = Rotation.from_matrix(np.eye(3)).as_quat()
    qend = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()
    rot_qs = np.linspace(qstart, qend, num_frames)

    tstart = np.zeros(3, np.float32)
    tend = world_to_camera[:3, 3]
    t_vecs = np.linspace(tstart, tend, num_frames)

    camera_start = [-7, 0, 2.5]
    lookat = [0, 0, 2.5]
    fov = 50
    view_cam = sp.Camera(camera_start, lookat, fov_y_degrees=fov)

    def _add_meshes(frame: sp.Frame3D, model_transform: np.ndarray,
                    camera: sp.Camera):
        for mesh in model:
            frame.add_mesh(mesh, transform=model_transform)

        frame.add_mesh(frustum, transform=world_to_camera)
        frame.add_mesh(camera_image, transform=world_to_camera)
        frame.add_mesh(coord_axes)

        pos = np.array([0, -.9, 1, 1], np.float32)
        pos = np.linalg.inv(camera.projection) @ pos
        frame.add_label(title, pos[:3])
        frame.add_label(uvw, [0, 0.6, 1])
        pos = np.array([0, -1.4, 0, 1], np.float32)
        pos = model_transform @ pos
        frame.add_label(xyz, pos[:3])

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

    camera_angles = np.concatenate([
        np.linspace(0, np.pi / 4, num_frames // 3),
        np.linspace(np.pi / 4, np.pi / 4, num_frames - num_frames // 3)
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

            t_start = t_values[first_leaf]
            t_end = t_start - (i * t_start) / (num_frames - 1)
            start = origin + t_start * direction
            end = origin + t_end * direction
            color = leaf_colors[path[first_leaf]]
            ray_mesh.add_thickline(color, start, end, 0.01, 0.01)

        ray_meshes.append(ray_mesh)

    for i in range(num_frames):
        frame = canvas.create_frame()
        frame.add_mesh(ray_meshes[i], transform=world_to_camera)
        rotation = sp.Transforms.rotation_matrix_from_axis_angle([0, 1, 0], camera_angles[i])
        position = np.ones(4, np.float32)
        position[:3] = camera_start
        position = rotation @ position
        frame.camera = sp.Camera(position[:3], lookat, fov_y_degrees=fov)
        _add_meshes(frame, world_to_camera, frame.camera)

    camera_angles = np.concatenate([
        np.linspace(np.pi / 4, -np.pi / 4, num_frames),
        np.linspace(-np.pi / 4, 0, num_frames)
    ])

    camera_fov = np.concatenate([
        np.linspace(fov, 40, num_frames),
        np.linspace(40, fov, num_frames)
    ])

    for i in range(num_frames * 2):
        frame = canvas.create_frame()
        frame.add_mesh(ray_meshes[-1], transform=world_to_camera)
        rotation = sp.Transforms.rotation_matrix_from_axis_angle([0, 1, 0], camera_angles[i])
        position = np.ones(4, np.float32)
        position[:3] = camera_start
        position = rotation @ position
        frame.camera = sp.Camera(position[:3], lookat, fov_y_degrees=camera_fov[i])
        _add_meshes(frame, world_to_camera, frame.camera)


if __name__ == "__main__":
    dataset = ffn.RayDataset.load("antinous_400.npz", "train", 64, True, False)
    voxels = ffn.OcTree.load("antinous_octree_8.npz")
    scene = sp.Scene()
    world_to_camera(scene, voxels, dataset.cameras[6], dataset.images[6], 800)
    print("Writing scenepic to file...")
    scene.save_as_html("world_to_camera.html", "World to Camera")
