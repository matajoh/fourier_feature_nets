"""Animation of a model increasing in voxel resolution."""

import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp


def voxels_animation(voxels: ffn.OcTree, min_depth=4, num_frames=300,
                     up_dir=(0, 1, 0), forward_dir=(0, 0, -1),
                     fov_y_degrees=40, resolution=(400, 400),
                     distance=4) -> sp.Scene:
    """Creates an animation of a model increasing in voxel resolution.

    Args:
        voxels (ffn.OcTree): The model. Needs to start at maximum resolution.
                              It will be pruned down to `min_depth`.
        min_depth (int, optional): Minimum voxel depth for the animation.
                                   Defaults to 4.
        num_frames (int, optional): Number of frames to animate.
                                    Defaults to 300.
        up_dir (tuple, optional): The up direction. Used to determine the
                                  axes of rotation for the animation.
                                  Defaults to (0, 1, 0).
        forward_dir (tuple, optional): The forward direction. Used to determine
                                       the starting position of the camera
                                       and the axes of rotation.
                                       Defaults to (0, 0, -1).
        fov_y_degrees (int, optional): Field of view for the camera.
                                       Defaults to 40.
        resolution (tuple, optional): Width and height of the canvas.
                                      Defaults to (400, 400).
        distance (int, optional): Camera distance. Defaults to 4.

    Returns:
        sp.Scene: A scene containing the animation.
    """
    up_dir = np.array(up_dir, np.float32)
    forward_dir = np.array(forward_dir, np.float32)
    resolution = ffn.Resolution(*resolution)

    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=resolution.width,
                                    height=resolution.height)
    canvas.shading = sp.Shading(sp.Colors.White)

    meshes = {}
    labels = {}
    max_depth = voxels.depth
    bar = ffn.ETABar("Pruning OcTree", max=voxels.depth - min_depth + 1)
    while voxels.depth >= min_depth:
        bar.next()
        bar.info(str(voxels.depth))
        depth_meshes = []
        leaf_centers = voxels.leaf_centers()
        leaf_depths = voxels.leaf_depths()
        leaf_colors = voxels.leaf_data()
        depths = np.unique(leaf_depths)
        for depth in depths:
            mesh = scene.create_mesh()
            transform = sp.Transforms.scale(pow(2., 1-depth) * voxels.scale)
            mesh.add_cube(sp.Colors.White, transform=transform)
            depth_centers = leaf_centers[leaf_depths == depth]
            depth_colors = leaf_colors[leaf_depths == depth]
            mesh.enable_instancing(depth_centers, colors=depth_colors)
            depth_meshes.append(mesh)

        meshes[voxels.depth] = depth_meshes
        text = "{} voxels".format(len(leaf_colors))
        labels[voxels.depth] = scene.create_label(text=text,
                                                  color=sp.Colors.Black,
                                                  font_family="arial",
                                                  size_in_pixels=75,
                                                  camera_space=True)
        voxels = voxels.prune()

    bar.finish()

    orbit_cameras = ffn.orbit(up_dir, forward_dir, num_frames,
                              fov_y_degrees, resolution, distance)
    sp_cameras = [cam.to_scenepic() for cam in orbit_cameras]

    frame_depth = np.linspace(min_depth, max_depth + 1,
                              num_frames, endpoint=False).astype(np.int32)
    for camera, depth in zip(sp_cameras, frame_depth):
        frame = canvas.create_frame()
        for mesh in meshes[depth]:
            frame.add_mesh(mesh)

        frame.add_label(labels[depth], [-.43, -.33, -1])
        frame.camera = camera

    return scene


if __name__ == "__main__":
    voxels = ffn.OcTree.load("antinous_octree_10.npz")
    scene = voxels_animation(voxels, resolution=(800, 800))
    scene.save_as_html("voxels_animation.html", "Voxels Animation")
