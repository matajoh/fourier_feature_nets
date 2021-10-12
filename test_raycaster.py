"""Integration test for the Raycaster (produces a cool scenepic)."""
from typing import List

from matplotlib.pyplot import get_cmap
from nerf import CameraInfo, OcTree
import numpy as np
import scenepic as sp


def _test_cameras() -> List[CameraInfo]:
    num_cameras, height, width = 10, 256, 256
    angle_diff = 2 * np.pi / num_cameras
    elevation = -np.pi / 6
    cameras = []
    for i in range(num_cameras):
        position = np.array([0, 0, 3, 1], np.float32)
        position = sp.Transforms.rotation_about_x(elevation) @ position
        position = sp.Transforms.rotation_about_y(i * angle_diff) @ position
        position = sp.Transforms.translate([0, -1, 0]) @ position
        position = position[:3]
        rotation = sp.Transforms.look_at_rotation([0, 0, 0], position, [0, -1, 0])
        rotation = np.linalg.inv(rotation)
        camera_to_world = sp.Transforms.translate(position) @ rotation
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = width
        intrinsics[0, 2] = width / 2
        intrinsics[1, 1] = width
        intrinsics[1, 2] = height / 2
        cameras.append(CameraInfo.create("cam{}".format(i), (width, height),
                                         intrinsics, camera_to_world))

    return cameras


def _main():
    cameras = _test_cameras()

    voxels = OcTree(4, 4096)
    positions = voxels.leaf_centers()
    opacity = np.random.uniform(0, 1, len(positions))
    voxels.split(opacity, 0.1)
    opacity = np.random.uniform(0, 1, len(voxels.leaves))
    voxels.merge(opacity, 0.1)
    opacity = np.random.uniform(0, 1, len(voxels.leaves))
    voxels.split(opacity, 0.1)
    positions = voxels.leaf_centers()

    pos_per_camera = 10
    pos_index = np.arange(len(positions))
    np.random.shuffle(pos_index)
    pos_index = pos_index[:pos_per_camera*len(cameras)]

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=600, height=600)

    voxel_mesh = scene.create_mesh(layer_id="voxels")
    for pos, scale in zip(voxels.leaf_centers(), voxels.leaf_scales()):
        transform = sp.Transforms.translate(pos) @ sp.Transforms.scale(scale * 2)
        voxel_mesh.add_cube(sp.Colors.White, transform=transform,
                            fill_triangles=False, add_wireframe=True)

    cmap = get_cmap("jet")
    colors = cmap(np.linspace(0, 1, len(pos_index)))[:, :3]
    for start in range(0, len(pos_index), pos_per_camera):
        end = min(start + pos_per_camera, len(pos_index))
        camera = cameras[start // pos_per_camera]
        camera_pos = pos_index[start:end]
        points = camera.project(positions[camera_pos])
        points, dirs = camera.raycast(points)

        batch_voxels = voxels.batch_intersect(points, dirs)

        camera_mesh = scene.create_mesh(layer_id="cameras")
        sp_camera = camera.to_scenepic()
        camera_mesh.add_camera_frustum(sp_camera, colors[start])

        for i, color in enumerate(colors[start:end]):
            t_stops = batch_voxels.t_stops[:, i]
            node_ids = batch_voxels.nodes[:, i]
            leaf_index = batch_voxels.leaf_index[:, i]
            t_stops = t_stops[leaf_index != -1]
            node_ids = node_ids[leaf_index != -1]

            ray_voxel_mesh = scene.create_mesh(layer_id="ray_voxels")
            ray_mesh = scene.create_mesh(layer_id="rays")
            t_last = 0.85
            for t, node_id in zip(t_stops, node_ids):
                node = voxels.nodes[node_id]
                voxel_transform = sp.Transforms.scale(2 * node.scale)
                voxel_transform = voxel_transform @ sp.Transforms.translate(node.center)
                ray_voxel_mesh.add_cube(color, transform=voxel_transform)
                p0 = points[i] + t_last * dirs[i]
                p1 = points[i] + t * dirs[i]
                ray_mesh.add_thickline(sp.Colors.Magenta, p0, p1,
                                       start_thickness=0.02, end_thickness=0.02)
                t_last = t

            frame = canvas.create_frame()
            frame.camera = sp_camera
            frame.add_mesh(ray_voxel_mesh)
            frame.add_mesh(camera_mesh)
            frame.add_mesh(ray_mesh)
            frame.add_mesh(voxel_mesh)

    scene.framerate = 10
    scene.save_as_html("raycast.html")


if __name__ == "__main__":
    _main()
