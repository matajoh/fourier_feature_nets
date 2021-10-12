# 1. project voxel centers into the images
# 2. ray cast them into the image
# 3. integrate along the ray (per voxel)

from operator import pos
from typing import List

from nerf2d import CameraInfo, OcTree

from matplotlib.pyplot import get_cmap
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
        rotation = np.linalg.inv(sp.Transforms.look_at_rotation([0, 0, 0], position, [0, -1, 0]))
        camera_to_world = sp.Transforms.translate(position) @ rotation
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = width
        intrinsics[0, 2] = width / 2
        intrinsics[1, 1] = width
        intrinsics[1, 2] = height / 2
        cameras.append(CameraInfo.create("cam{}".format(i), (width, height), intrinsics, camera_to_world))
    
    return cameras


def _main():
    cameras = _test_cameras()

    voxels = OcTree(4, 4096)
    positions = voxels.leaf_centers()
    opacity = np.random.uniform(0, 1, len(positions))
    opacity = voxels.merge(opacity, 0.1)
    opacity = voxels.merge(opacity, 0.1)
    opacity = voxels.split(opacity, 0.1)
    voxels.split(opacity, 0.1)

    pos_index = np.arange(positions.shape[0])
    np.random.shuffle(pos_index)
    pos_per_camera = (len(pos_index) // len(cameras)) + 1

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=600, height=600)

    voxel_mesh = scene.create_mesh(layer_id="voxels")
    for pos, scale in zip(voxels.leaf_centers(), voxels.leaf_scales()):
        transform = sp.Transforms.translate(pos) @ sp.Transforms.scale(scale * 2)
        voxel_mesh.add_cube(sp.Colors.White, transform=transform, fill_triangles=False, add_wireframe=True)

    cmap = get_cmap("jet")
    colors = cmap(np.linspace(0, 1, len(positions)))[:, :3]
    for start in range(0, len(pos_index), pos_per_camera):
        end = min(start + pos_per_camera, len(pos_index))
        camera = cameras[start // pos_per_camera]
        camera_pos = pos_index[start:end]
        points = camera.project(positions[camera_pos])
        points, dirs = camera.raycast(points)

        ray_voxels = voxels.batch_intersect(points, dirs)

        camera_mesh = scene.create_mesh(layer_id="cameras")
        sp_camera = camera.to_scenepic()
        camera_mesh.add_camera_frustum(sp_camera, colors[start])

        for point, dir, color in zip(points, dirs, colors[start:end]):
            ray_voxel_mesh = scene.create_mesh(layer_id="ray_voxels")
            ray_mesh = scene.create_mesh(layer_id="rays")
            ray_voxels = voxels.intersect(point, dir)
            t_last = 0.85
            for t, node_id in ray_voxels:
                node = voxels.nodes[node_id]
                ray_voxel_mesh.add_cube(color, transform=sp.Transforms.translate(node.center) @ sp.Transforms.scale(2 * node.scale))
                p0 = point + t_last * dir
                p1 = point + t * dir
                ray_mesh.add_thickline(sp.Colors.Magenta, p0, p1, start_thickness=0.02, end_thickness=0.02)
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
    