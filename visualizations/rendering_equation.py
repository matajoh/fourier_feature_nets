"""Animation of the rendering equation."""

import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp
import torch
import torch.nn.functional as F


def rendering_equation(voxels: ffn.OcTree, ray_samples: ffn.RaySamples,
                       camera: ffn.CameraInfo, image: np.ndarray,
                       model: ffn.NeRF, resolution=800) -> sp.Scene:
    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=resolution, height=3 * resolution / 4)
    canvas.shading = sp.Shading(bg_color=sp.Colors.White)

    graph = scene.create_graph(width=resolution, height=resolution / 4,
                               text_size=24)

    num_samples = len(ray_samples.positions[0])

    leaf_centers = voxels.leaf_centers()
    leaf_depths = voxels.leaf_depths()
    leaf_colors = voxels.leaf_data()
    depths = np.unique(leaf_depths)
    cubes = []
    for depth in depths:
        mesh = scene.create_mesh(layer_id="model")
        transform = sp.Transforms.scale(pow(2., 1-depth) * voxels.scale)
        mesh.add_cube(sp.Colors.White, transform=transform, add_wireframe=True, fill_triangles=False)
        depth_centers = leaf_centers[leaf_depths == depth]
        depth_colors = leaf_colors[leaf_depths == depth]
        mesh.enable_instancing(depth_centers, colors=depth_colors)
        cubes.append(mesh)

    sp_image = scene.create_image()
    image = image[..., :3]
    sp_image.from_numpy(image)
    camera_image = scene.create_mesh(texture_id=sp_image.image_id,
                                     double_sided=True)
    camera_image.add_camera_image(camera.to_scenepic())

    frustum = scene.create_mesh()
    frustum.add_camera_frustum(camera.to_scenepic(), sp.Colors.White)

    positions = ray_samples.positions.reshape(-1, 3)
    views = ray_samples.view_directions.reshape(-1, 3)
    model.eval()
    with torch.no_grad():
        color_o = model(positions, views)

    color_o = color_o.reshape(1, num_samples, 4)
    color, opacity = torch.split(color_o, [3, 1], -1)
    color = torch.sigmoid(color)
    opacity = F.softplus(opacity)

    assert not color.isnan().any()
    assert not opacity.isnan().any()

    opacity = opacity.squeeze(-1)
    deltas = ray_samples.t_values[:, 1:] - ray_samples.t_values[:, :-1]
    max_dist = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, max_dist], dim=-1)
    trans = torch.exp(-(opacity * deltas).cumsum(-1))
    
    graph.add_sparkline("σ", opacity[0].numpy(), sp.Colors.Red, 2)
    graph.add_sparkline("T", trans[0].numpy(), sp.Colors.Blue, 2)

    camera_start = [0, 0, 1.5]
    lookat = [0, 0, 0]
    fov = 70

    def _add_meshes(frame: sp.Frame3D):
        for mesh in cubes:
            frame.add_mesh(mesh)

        frame.add_mesh(frustum)
        frame.add_mesh(camera_image)

    for i in range(num_samples):
        start = camera.position
        end = ray_samples.positions[0, i]
        ray_mesh = scene.create_mesh()
        ray_mesh.add_thickline(sp.Colors.Black, start, end, 0.005, 0.005)

        if i < num_samples / 2:
            angle = (i * np.pi) / num_samples
        else:
            angle = ((num_samples - i) * np.pi) / num_samples

        view_rot = sp.Transforms.rotation_about_y(angle)
        view_pos = view_rot[:3, :3] @ np.array(camera_start)
        view_cam = sp.Camera(view_pos, lookat, fov_y_degrees=fov, aspect_ratio=4/3)

        sample_mesh = scene.create_mesh()
        sample_mesh.add_sphere(sp.Colors.White, transform=sp.Transforms.scale(0.03))
        positions = ray_samples.positions[0, :i+1]
        colors = color[0, :i+1]
        index = opacity[0, :i+1] > 0.1
        positions = positions[index]
        colors = colors[index]
        sample_mesh.enable_instancing(positions.numpy(), colors=colors.numpy())

        frame = canvas.create_frame(camera=view_cam)
        _add_meshes(frame)
        frame.add_mesh(ray_mesh)
        frame.add_mesh(sample_mesh)

    scene.link_canvas_events(canvas, graph)
    return scene


if __name__ == "__main__":
    voxels = ffn.OcTree.load("antinous_octree_7.npz")
    dataset = ffn.ImageDataset.load("antinous_400.npz", "train", 256, True, True)
    model = ffn.load_model("antinous_800_nerf.pt")
    image = dataset.images[17]
    row = 190
    col = 240
    camera = dataset.cameras[17]
    index = 17 * dataset.sampler.rays_per_camera + 190 * 400 + 240
    rays = dataset.sampler.sample([index], None)
    scene = rendering_equation(voxels, rays, camera, image, model)
    scene.save_as_html("rendering_eq.html", title="Rendering Equation")
