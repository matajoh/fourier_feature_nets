from typing import List, NamedTuple, Tuple

import fourier_feature_nets as ffn
import numpy as np
import scenepic as sp


class Ray(NamedTuple("Ray", [("x", float), ("y", float), ("z", float),
                             ("dx", float), ("dy", float), ("dz", float)])):
    def cast(self, t: float) -> np.ndarray:
        x = self.x + t * self.dx
        y = self.y + t * self.dy
        z = self.z + t * self.dz
        return np.array([x, y, z], np.float32)


Intersection = NamedTuple("Intersection", [("enter", float), ("exit", float)])


def _in_order(a: float, b: float) -> Tuple[float, float]:
    if b < a:
        return b, a

    return a, b


def _near_far(coord_diff: float, ray_dir: float):
    near = (coord_diff - 1) / ray_dir
    far = (coord_diff + 1) / ray_dir
    return _in_order(near, far)


def _min(x: float, y: float, z: float) -> Tuple[float, int]:
    if x < y:
        if x < z:
            return x, 0
    else:
        if y < z:
            return y, 1

    return z, 2


def _max(x: float, y: float, z: float) -> Tuple[float, int]:
    if x > y:
        if x > z:
            return x, 0
    else:
        if y > z:
            return y, 1

    return z, 2


def _intersect_cube_with_ray(ray: Ray) -> List[Intersection]:
    x0, x1 = _near_far(-ray.x, ray.dx)
    y0, y1 = _near_far(-ray.y, ray.dy)
    z0, z1 = _near_far(-ray.z, ray.dz)

    return [Intersection(x0, x1),
            Intersection(y0, y1),
            Intersection(z0, z1)]


def _random_point() -> np.ndarray:
    point = np.random.random_sample(size=3) + 1
    sign = np.sign(np.random.random_sample(size=3) - 0.5)
    return (point * sign).astype(np.float32)


def _on_edge(x: float) -> bool:
    if x > 0:
        return abs(x - 1) < 1e-2

    return abs(x + 1) < 1e-2


def _build_animation(num_rays: int, num_samples, num_pause) -> sp.Scene:
    scene = sp.Scene()
    main = scene.create_canvas_3d("main", width=600, height=600)
    main.shading = sp.Shading(bg_color=sp.Colors.White)
    x_proj = scene.create_canvas_2d("x_proj", width=200, height=200,
                                    background_color=sp.Colors.White)
    y_proj = scene.create_canvas_2d("y_proj", width=200, height=200,
                                    background_color=sp.Colors.White)
    z_proj = scene.create_canvas_2d("z_proj", width=200, height=200,
                                    background_color=sp.Colors.White)

    cube_mesh = scene.create_mesh("cube")
    cube_mesh.add_cube(sp.Colors.Black, transform=sp.Transforms.scale(2),
                       add_wireframe=True, fill_triangles=False)
    cube_mesh.add_coordinate_axes(transform=sp.Transforms.scale(0.5))

    up_dir = np.array([0, 1, 0], np.float32)
    forward_dir = np.array([0, 0, 1], np.float32)
    orbit = ffn.orbit(up_dir, forward_dir, num_rays * (num_samples + 2*num_pause),
                      65, ffn.Resolution(600, 600), 5)
    orbit = iter(orbit)

    for _ in range(num_rays):
        ray_start = _random_point()
        ray_end = _random_point()
        check = ray_start * ray_end
        if (check > 0).any():
            index = np.nonzero(check > 0)
            ray_end[index] *= -1

        direction = (ray_end - ray_start)
        length = np.linalg.norm(direction)
        direction /= length
        ray = Ray(*ray_start, *direction)
        x_int, y_int, z_int = _intersect_cube_with_ray(ray)
        samples = np.linspace(0, length, num_samples)
        t_min, a_min = _max(x_int.enter, y_int.enter, z_int.enter)
        t_max, a_max = _min(x_int.exit, y_int.exit, z_int.exit)
        colors = [sp.Colors.Red, sp.Colors.Green, sp.Colors.Blue]
        samples = np.sort(np.concatenate([samples, np.array([t_min, t_max])]))

        for sample in samples:
            ray_mesh = scene.create_mesh()
            point = ray.cast(sample)
            ray_mesh.add_thickline(sp.Colors.Black, ray_start, point, 0.01, 0.01)

            transform = sp.Transforms.scale(0.15)
            transform = sp.Transforms.translate(point) @ transform
            num_frames = 1
            if sample == t_min:
                num_frames = num_pause
                ray_mesh.add_sphere(colors[a_min], transform=transform)
            elif sample == t_max:
                num_frames = num_pause
                ray_mesh.add_sphere(colors[a_max], transform=transform)

            coords = np.stack([ray_start, point])

            for _ in range(num_frames):
                camera = next(orbit).to_scenepic()
                main.create_frame(meshes=[cube_mesh, ray_mesh], camera=camera)

                for axis, proj in enumerate([x_proj, y_proj, z_proj]):
                    frame = proj.create_frame()
                    frame.add_rectangle(400/6, 400/6, 400/6, 400/6, colors[axis], 2)
                    coords2d = np.roll(coords, axis, axis=1)[:, 1:]
                    coords2d[:, 1] *= -1
                    x, y = coords2d[-1]
                    coords2d = (coords2d + 3) * 200 / 6
                    if sample == t_min and (_on_edge(x) or _on_edge(y)):
                        frame.add_circle(*coords2d[-1], 4, fill_color=colors[a_min])
                    elif sample == t_max and (_on_edge(x) or _on_edge(y)):
                        frame.add_circle(*coords2d[-1], 4, fill_color=colors[a_max])

                    frame.add_line(coords2d, line_width=2)

    scene.grid("800px", "200px 200px 200px", "600px 200px")
    scene.place(main.canvas_id, "1 / span 3", "1")
    scene.place(x_proj.canvas_id, "1", "2")
    scene.place(y_proj.canvas_id, "2", "2")
    scene.place(z_proj.canvas_id, "3", "2")
    scene.link_canvas_events(main, x_proj, y_proj, z_proj)
    return scene


if __name__ == "__main__":
    scene = _build_animation(5, 100, 20)
    scene.save_as_html("ray_cube_int.html", title="Ray/Cube Intersection")
