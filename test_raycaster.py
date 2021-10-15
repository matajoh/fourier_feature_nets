"""Integration test for the Raycaster (produces a cool scenepic)."""
import timeit
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

    voxels = OcTree(4)
    positions = voxels.leaf_centers()
    opacity = np.random.uniform(0, 1, voxels.num_leaves)
    voxels.split(opacity, 0.1, 4096)
    opacity = np.random.uniform(0, 1, voxels.num_leaves)
    voxels.merge(opacity, 0.1)
    opacity = np.random.uniform(0, 1, voxels.num_leaves)
    voxels.split(opacity, 0.1, 4096)
    positions = voxels.leaf_centers()
    scales = voxels.leaf_scales()

    pos_per_camera = 10
    pos_index = np.arange(len(positions))
    np.random.shuffle(pos_index)
    pos_index = pos_index[:pos_per_camera*len(cameras)]

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=600, height=600)

    voxel_mesh = scene.create_mesh(layer_id="voxels")
    for pos, scale in zip(positions, scales):
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
        starts, dirs = camera.raycast(points)

        time_taken = timeit.timeit(lambda: voxels.intersect(starts, dirs, 10), number=3)
        print("Best of 3:", time_taken, "s")

        ray_paths = voxels.intersect(starts, dirs, 10)

        camera_mesh = scene.create_mesh(layer_id="cameras")
        sp_camera = camera.to_scenepic()
        camera_mesh.add_camera_frustum(sp_camera, colors[start])

        for i, color in enumerate(colors[start:end]):
            ray_voxel_mesh = scene.create_mesh(layer_id="ray_voxels")
            ray_mesh = scene.create_mesh(layer_id="rays")
            t_last = 0.85
            for t, leaf in ray_paths[i]:
                voxel_transform = sp.Transforms.scale(2 * scales[leaf])
                voxel_transform = sp.Transforms.translate(positions[leaf]) @ voxel_transform
                ray_voxel_mesh.add_cube(color, transform=voxel_transform)
                p0 = starts[i] + t_last * dirs[i]
                p1 = starts[i] + t * dirs[i]
                ray_mesh.add_thickline(sp.Colors.Magenta, p0, p1,
                                       start_thickness=0.02, end_thickness=0.02)
                t_last = t

            frame = canvas.create_frame()
            frame.camera = sp_camera
            frame.add_mesh(ray_voxel_mesh)
            frame.add_mesh(camera_mesh)
            frame.add_mesh(ray_mesh)
            frame.add_mesh(voxel_mesh)

    scene.framerate = 5
    scene.save_as_html("voxel_raycaster.html", "Voxel Raycaster")


if __name__ == "__main__":
    _main()
