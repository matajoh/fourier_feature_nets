"""Module containing an optimized implementation of an OcTree."""

from collections import deque
import os
from typing import Dict, List, NamedTuple, Set, Tuple, Union

from numba import njit
import numpy as np
from progress.bar import ChargingBar
import trimesh

from .utils import ETABar, download_asset, interpolate_bilinear


Vector = NamedTuple("Vector", [("x", float), ("y", float), ("z", float)])
Node = NamedTuple("Node", [("id", int),
                           ("x", float), ("y", float), ("z", float),
                           ("scale", float), ("depth", int)])
Ray = NamedTuple("Ray", [("x", float), ("y", float), ("z", float),
                         ("dx", float), ("dy", float), ("dz", float)])
Intersection = NamedTuple("Intersection", [("t_min", float), ("a_min", int),
                                           ("t_max", float), ("a_max", int)])
Path = NamedTuple("Path", [("t_stops", np.ndarray), ("leaves", np.ndarray)])
NodePriority = NamedTuple("NodePriority", [("depth", int), ("std", float),
                                           ("opacity", float), ("node", Node)])


@njit
def _corput(index, base):
    """Generate the nth Van der Corput number."""
    x = 0
    norm = 1.0/base

    while index > 0:
        x += (index % base)*norm
        index //= base
        norm /= base

    return x


@njit
def _sample_regular_barys(points_per_triangle):
    """Algorithm from Basu and Owen, 'Low discrepancy constructions in the triangle'."""
    max_corput = np.max(points_per_triangle)
    corput_values = np.array([_corput(i + 1, 4)
                              for i in range(max_corput)], np.float32)
    num_points = sum(points_per_triangle)
    samples = np.zeros(num_points, np.float32)
    start = 0
    for count in points_per_triangle:
        end = start + count
        samples[start:end] = corput_values[:count]
        start = end

    a = np.zeros((num_points, 2), np.float32)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    a[:, 0] = 1
    b[:, 1] = 1
    a_new = np.zeros_like(a)
    b_new = np.zeros_like(b)
    c_new = np.zeros_like(c)
    samples = (samples * (1 << 32)).astype(np.uint32)
    for i in range(16):
        print("Iteration", i + 1, "of 16")
        a_new[:] = 0
        b_new[:] = 0
        c_new[:] = 0

        d = (samples >> (2 * (15 - i))) & 0x3
        idx = d == 0
        a_new[idx] = (b[idx] + c[idx]) / 2
        b_new[idx] = (a[idx] + c[idx]) / 2
        c_new[idx] = (a[idx] + b[idx]) / 2

        idx = d == 1
        a_new[idx] = a[idx]
        b_new[idx] = (a[idx] + b[idx]) / 2
        c_new[idx] = (a[idx] + c[idx]) / 2

        idx = d == 2
        a_new[idx] = (b[idx] + a[idx]) / 2
        b_new[idx] = b[idx]
        c_new[idx] = (b[idx] + c[idx]) / 2

        idx = d == 3
        a_new[idx] = (c[idx] + a[idx]) / 2
        b_new[idx] = (c[idx] + b[idx]) / 2
        c_new[idx] = c[idx]

        a, a_new = a_new, a
        b, b_new = b_new, b
        c, c_new = c_new, c

    barys = np.zeros((num_points, 3), np.float32)
    barys[:, :2] = (a + b + c) / 3
    barys[:, 2] = 1 - np.sum(barys, axis=1)
    return barys


@njit
def _barycentric_interpolation(barycentric_ids,
                               barycentric_coords,
                               vertex_function):
    """Interpolates points within triangles."""
    num_samples = barycentric_ids.shape[0]
    dim = vertex_function.shape[-1]
    vertex_values = np.zeros((num_samples, 3, dim), vertex_function.dtype)
    for i in range(num_samples):
        index = barycentric_ids[i]
        vertex_values[i] = vertex_function[index]

    barycentric_coords = barycentric_coords.reshape(num_samples, 3, 1)
    values = np.multiply(vertex_values, barycentric_coords)
    values = np.sum(values, -2)
    return values


def _sample_barycentric_point_cloud(vertex_positions, triangles,
                                    uvs, num_points):
    """Sample a point cloud from a mesh using barycentric coordinates."""
    triangle_verts = vertex_positions[triangles]
    normals = np.cross(triangle_verts[:, 2] - triangle_verts[:, 0],
                       triangle_verts[:, 1] - triangle_verts[:, 0])
    area = 0.5*np.linalg.norm(normals, axis=-1)
    area = area / area.sum()
    sample_indices = np.random.choice(len(area), size=num_points, p=area)
    counts = np.bincount(sample_indices, minlength=len(triangles))
    bary_ids = np.repeat(np.arange(len(triangles)), counts).astype(np.int64)
    bary_ids = triangles[bary_ids]
    bary_coords = _sample_regular_barys(counts)
    sample_verts = _barycentric_interpolation(bary_ids, bary_coords,
                                              vertex_positions)
    sample_uvs = _barycentric_interpolation(bary_ids, bary_coords, uvs)
    return sample_verts, sample_uvs


# skew-symmetric cross-product
# this matrix extracts the values from
# the vector into a 9d vector that can
# be reshaped as a 3x3 matrix
TO_SSCP_MATRIX = np.array([
    [0, 0, 0], [0, 0, -1], [0, 1, 0],
    [0, 0, 1], [0, 0, 0], [-1, 0, 0],
    [0, -1, 0], [1, 0, 0], [0, 0, 0]
], np.float32)


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Performs the vector dot product along the last dimension."""
    return (x * y).sum(-1)


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produces the rotation needed to align two vectors."""
    v = np.cross(a, b).reshape(3, 1)
    cos = _dot(a, b).reshape(1, 1)
    vx = np.matmul(TO_SSCP_MATRIX, v)
    vx = vx.reshape(3, 3)
    vxvx = np.matmul(vx, vx)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] += vx + (1 / (1 + cos)) * vxvx
    return transform


def _transform_points(transform: np.ndarray, points: np.ndarray):
    """Transforms points using homogeneous coordinates."""
    h_points = np.concatenate([points, np.ones_like(points[:, :1])], -1)
    h_points = (transform @ h_points.T).T
    return h_points[:, :3]


def _normalize_points(vertex_positions, up_dir):
    # rotate to make up = y+
    transform = _align_vectors(np.array(up_dir), np.array([0, 1, 0]))
    translate = np.eye(4, dtype=np.float32)
    translate[:3, 3] = -vertex_positions.mean(0)
    transform = transform @ translate

    # scale to occupy volume
    vertex_positions = _transform_points(transform, vertex_positions)
    max_corner = vertex_positions.max(0)
    min_corner = vertex_positions.min(0)

    max_dim = (max_corner - min_corner).max()

    scale = 1.6 / max_dim
    vertex_positions *= scale

    # translate to fit entirely within volume
    max_corner = vertex_positions.max(0)
    min_corner = vertex_positions.min(0)
    center = 0.5 * (max_corner + min_corner)
    vertex_positions -= center

    return vertex_positions


@njit
def _in_order(a: float, b: float) -> Tuple[float, float]:
    if b < a:
        return b, a

    return a, b


@njit
def _near_far(coord_diff: float, scale: float, ray_dir: float):
    near = (coord_diff - scale) / ray_dir
    far = (coord_diff + scale) / ray_dir
    return _in_order(near, far)


@njit
def _min(x: float, y: float, z: float) -> Tuple[float, int]:
    if x < y:
        if x < z:
            return x, 0
    else:
        if y < z:
            return y, 1

    return z, 2


@njit
def _max(x: float, y: float, z: float) -> Tuple[float, int]:
    if x > y:
        if x > z:
            return x, 0
    else:
        if y > z:
            return y, 1

    return z, 2


@njit
def _intersect_node_with_ray(node: Node, ray: Ray) -> Intersection:
    x0, x1 = _near_far(node.x - ray.x, node.scale, ray.dx)
    y0, y1 = _near_far(node.y - ray.y, node.scale, ray.dy)
    z0, z1 = _near_far(node.z - ray.z, node.scale, ray.dz)

    t0, a0 = _max(x0, y0, z0)
    t1, a1 = _min(x1, y1, z1)
    return Intersection(t0, a0, t1, a1)


@njit
def _cast_ray(ray: Ray, t: float) -> Vector:
    x = ray.x + t * ray.dx
    y = ray.y + t * ray.dy
    z = ray.z + t * ray.dz
    return Vector(x, y, z)


@njit
def _node_contains(node: Node, point: Vector) -> bool:
    dx = abs(node.x - point.x)
    dy = abs(node.y - point.y)
    dz = abs(node.z - point.z)
    return not (dx > node.scale or dy > node.scale or dz > node.scale)


X_POS = 0b100
Y_POS = 0b010
Z_POS = 0b001
XY_MASK = 0b110
XZ_MASK = 0b101
YZ_MASK = 0b011


@njit
def _find_child_index_of_node(node: Node, point: Vector) -> Node:
    child_id = 0
    if point.x >= node.x:
        child_id += X_POS

    if point.y >= node.y:
        child_id += Y_POS

    if point.z >= node.z:
        child_id += Z_POS

    return child_id


@njit
def _find_child_of_node(node: Node, point: Vector) -> Node:
    scale = node.scale / 2
    child_id = (node.id << 3) + 1
    if point.x < node.x:
        x = node.x - scale
        if point.y < node.y:
            y = node.y - scale
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale, node.depth + 1)
            else:
                z = node.z + scale
                child_id += Z_POS
                return Node(child_id, x, y, z, scale, node.depth + 1)
        else:
            y = node.y + scale
            child_id += Y_POS
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale, node.depth + 1)
            else:
                z = node.z + scale
                child_id += Z_POS
                return Node(child_id, x, y, z, scale, node.depth + 1)
    else:
        x = node.x + scale
        child_id += X_POS
        if point.y < node.y:
            y = node.y - scale
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale, node.depth + 1)
            else:
                z = node.z + scale
                child_id += Z_POS
                return Node(child_id, x, y, z, scale, node.depth + 1)
        else:
            y = node.y + scale
            child_id += Y_POS
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale, node.depth + 1)
            else:
                z = node.z + scale
                child_id += Z_POS
                return Node(child_id, x, y, z, scale, node.depth + 1)


@njit
def _parent_of_node(node: Node) -> Node:
    parent_id = (node.id - 1) >> 3
    parent_scale = node.scale * 2
    start = (parent_id << 3) + 1
    child_id = node.id - start
    if child_id & X_POS:
        x = node.x - node.scale
    else:
        x = node.x + node.scale

    if child_id & Y_POS:
        y = node.y - node.scale
    else:
        y = node.y + node.scale

    if child_id & Z_POS:
        z = node.z - node.scale
    else:
        z = node.z + node.scale

    return Node(parent_id, x, y, z, parent_scale, node.depth - 1)


@njit
def _find_sibling_of_node(node: Node, point: Vector, axis: int) -> Node:
    parent_id = (node.id - 1) >> 3
    parent_scale = node.scale * 2
    start = (parent_id << 3) + 1
    child_id = node.id - start
    sibling_id = child_id
    if axis == 0:
        y = node.y
        z = node.z
        if child_id & X_POS:
            if point.x > node.x:
                return node

            sibling_id &= YZ_MASK
            x = node.x - parent_scale
        else:
            if point.x < node.x:
                return node

            sibling_id |= 4
            x = node.x + parent_scale
    elif axis == 1:
        x = node.x
        z = node.z
        if child_id & Y_POS:
            if point.y > node.y:
                return node

            sibling_id &= XZ_MASK
            y = node.y - parent_scale
        else:
            if point.y < node.y:
                return node

            sibling_id |= Y_POS
            y = node.y + parent_scale
    else:
        x = node.x
        y = node.y
        if child_id & Z_POS:
            if point.z > node.z:
                return node

            sibling_id &= XY_MASK
            z = node.z - parent_scale
        else:
            if point.z < node.z:
                return node

            sibling_id |= Z_POS
            z = node.z + parent_scale

    return Node(start + sibling_id, x, y, z, node.scale, node.depth)


@njit
def _trace_ray_path(scale: float, node_index: np.ndarray,
                    leaf_index: np.ndarray, start: np.ndarray,
                    direction: np.ndarray, max_length: int) -> Path:
    stack = [Node(0, 0.0, 0.0, 0.0, scale, 0)]
    ray = Ray(start[0], start[1], start[2],
              direction[0], direction[1], direction[2])
    tr = _intersect_node_with_ray(stack[0], ray)
    t = tr.t_min + 1e-5
    point = _cast_ray(ray, t)
    stop = 0
    t_stops = np.full(max_length, tr.t_max, np.float32)
    leaves = np.full(max_length, -1, np.int64)
    while stack:
        current = stack[-1]

        # need to keep searching down the tree
        index = np.searchsorted(node_index, current.id)
        if index < len(node_index) and node_index[index] == current.id:
            # we're in a valid node. Is the leaf somewhere in here?
            if _node_contains(current, point):
                # Yes. Push.
                child = _find_child_of_node(current, point)
                stack.append(child)
            else:
                # Nope. Pop.
                stack.pop()
        else:
            # Advance.
            tc = _intersect_node_with_ray(current, ray)
            t_stops[stop] = t

            index = np.searchsorted(leaf_index, current.id)
            if leaf_index[index] == current.id:
                # we're in a leaf.
                leaves[stop] = index
            else:
                # We're in empty space.
                leaves[stop] = -1

            stack.pop()
            stop += 1
            if t >= tr.t_max or stop == max_length - 1:
                # max path length reached, done.
                break

            # scootch along a bit beyond the t_max to ensure we're
            # in the next node.
            t = tc.t_max + 1e-5
            point = _cast_ray(ray, t)
            while _node_contains(current, point):
                # ...very paranoid about this failure case.
                # we NEED to leave the current leaf or the algorithm
                # will never return. 
                # TOD This would be safer/better/faster with integers.
                t += 1e-5
                point = _cast_ray(ray, t)

            # did we stumble into a sibling?
            sibling = _find_sibling_of_node(current, point, tc.a_max)
            if sibling != current:
                # nice, no need to pop back to the parent.
                stack.append(sibling)

    return Path(t_stops, leaves)


@njit
def _batch_intersect(scale: float,
                     node_index: np.ndarray,
                     leaf_index: np.ndarray,
                     starts: np.ndarray,
                     directions: np.ndarray,
                     max_length: int) -> Path:
    num_rays = len(starts)
    t_stops = np.zeros((num_rays, max_length), np.float32)
    leaves = np.zeros((num_rays, max_length), np.int64)
    for ray, (start, direction) in enumerate(zip(starts, directions)):
        path = _trace_ray_path(scale, node_index, leaf_index,
                               start, direction, max_length)
        t_stops[ray] = path.t_stops
        leaves[ray] = path.leaves

    return Path(t_stops, leaves)


@njit
def _batch_assign(node: Node, positions: np.ndarray) -> np.ndarray:
    result = np.zeros(len(positions), np.uint8)
    for i, (x, y, z) in enumerate(positions):
        result[i] = _find_child_index_of_node(node, Vector(x, y, z))

    return result


def _query(scale: float,
           node_index: np.ndarray,
           leaf_index: np.ndarray,
           point: Vector) -> int:
    node = Node(0, 0.0, 0.0, 0.0, scale, 0)
    if not _node_contains(node, point):
        return -1

    max_id = leaf_index[-1]
    while node.id <= max_id:
        node = _find_child_of_node(node, point)
        index = np.searchsorted(leaf_index, node.id)
        if leaf_index[index] == node.id:
            return index

        index = np.searchsorted(node_index, node.id)
        if index == len(node_index) or node_index[index] != node.id:
            return -1


def _batch_query(scale: float,
                 node_index: np.ndarray,
                 leaf_index: np.ndarray,
                 points: np.ndarray) -> np.ndarray:
    result = np.zeros(len(points), np.int64)
    for i, (x, y, z) in enumerate(points):
        result[i] = _query(scale, node_index, leaf_index, Vector(x, y, z))

    return result


@njit
def _list_children(node: Node) -> List[Node]:
    scale = node.scale / 2
    child_id = (node.id << 3) + 1
    depth = node.depth + 1
    x0 = node.x - scale
    y0 = node.y - scale
    z0 = node.z - scale
    x1 = node.x + scale
    y1 = node.y + scale
    z1 = node.z + scale
    nodes = []
    for x in [x0, x1]:
        for y in [y0, y1]:
            for z in [z0, z1]:
                child = Node(child_id, x, y, z, scale, depth)
                nodes.append(child)
                child_id += 1

    return nodes


def _leaf_nodes(scale: float, node_ids: Set[int], leaf_ids: Set[int]) -> List[Node]:
    queue = deque([Node(0, 0.0, 0.0, 0.0, scale, 0)])
    leaves = []
    report_interval = len(leaf_ids) // 100
    bar = ETABar("Building OcTree", max=len(leaf_ids))
    while queue:
        current = queue.popleft()
        if current.id in leaf_ids:
            leaves.append(current)
            if len(leaves) % report_interval == 0:
                bar.next(report_interval)
        elif current.id in node_ids:
            queue.extend(_list_children(current))

    bar.finish()
    return leaves


class OcTree:
    """Class representing an OcTree datastructure."""

    def __init__(self, scale: float, node_ids: Set[int], leaf_ids: Set[int],
                 leaf_data: np.ndarray = None):
        """Constructor.

        Description:
            All IDs are assumed to be in the format such that the child
            of a parent with ID i have IDs which start at (8*i) + 1
            and proceed linearly, for example:

            parent: 0
            children: 1 2 3 4 5 6 7 8
            grandchildren1: 9 10 11 12 13 14 15 16

        Args:
            scale (float): The scale of the root cube (measured as half the
                           side length, i.e. scale of 1 is a cube of side 2).
            node_ids (Set[int]): ID values of the nodes
            leaf_ids (Set[int]): ID values of the leaves
            leaf_data (np.ndarray, optional): [description]. Defaults to None.
        """
        self._update(node_ids, leaf_ids, scale)
        self._leaf_data = leaf_data

    def leaf_centers(self) -> np.ndarray:
        """The Nx3 center coordinates of all leaves."""
        leaf_centers = [[leaf.x, leaf.y, leaf.z] for leaf in self.leaves]
        return np.array(leaf_centers, np.float32)

    def leaf_depths(self) -> np.ndarray:
        """The N depths for all leaves."""
        leaf_depths = [leaf.depth for leaf in self.leaves]
        return np.array(leaf_depths, np.int32)

    def leaf_data(self) -> np.ndarray:
        """The data stored in each leaf."""
        return self._leaf_data

    @property
    def depth(self) -> int:
        """The maximum depth of the tree."""
        node_id = self._leaf_index[-1].item()
        depth = 0
        while node_id > 0:
            node_id = (node_id - 1) >> 3
            depth += 1

        return depth + 1

    def prune(self) -> "OcTree":
        """Prunes all leaves at the maximum depth."""
        if self._leaf_data is None:
            leaf_data = np.zeros((len(self.leaves), 1))
            no_data = True
        else:
            leaf_data = self._leaf_data
            no_data = False

        # we want to remove all leaves at maximum depth:
        # 1. merge max_depth leaves into their parent nodes
        # 2. make the resulting parent nodes the new leaves
        max_depth = self.depth - 1
        node_ids = set(self._node_index.tolist())
        new_leaf_data = {}
        new_leaf_counts = {}
        for leaf, data in zip(self.leaves, leaf_data):
            if leaf.depth < max_depth:
                # perfectly fine, gets to stay
                new_leaf_data[leaf.id] = data
                new_leaf_counts[leaf.id] = 1
                continue

            parent = _parent_of_node(leaf)
            if parent.id not in new_leaf_data:
                node_ids.remove(parent.id)
                new_leaf_data[parent.id] = np.zeros_like(data)
                new_leaf_counts[parent.id] = 0

            new_leaf_data[parent.id] += data
            new_leaf_counts[parent.id] += 1

        leaf_ids = list(sorted(new_leaf_data.keys()))
        leaf_data = [new_leaf_data[i] / new_leaf_counts[i] for i in leaf_ids]
        leaf_data = None if no_data else np.stack(leaf_data)
        leaf_ids = set(leaf_ids)
        return OcTree(self._scale, node_ids, leaf_ids, leaf_data)

    def __len__(self):
        """Counts all the nodes in the tree."""
        return len(self._node_ids) + len(self._leaf_ids)

    @property
    def num_leaves(self) -> int:
        """Counts the number of leaves in the tree."""
        return len(self._leaf_ids)

    @property
    def scale(self) -> float:
        """Scale of the cube (side is 2 * scale)."""
        return self._scale

    def query(self, positions: np.ndarray) -> np.ndarray:
        """Performs a query into the OcTree for the provided positions.

        Args:
            positions (np.ndarray): a (N,3) tensor of positions

        Returns:
            np.ndarray: a (N) tensor of leaf IDs for the leaves which contain
            the positions. -1 indicates the position is out of bounds or in an
            empty node.
        """
        assert positions.shape[-1] == 3
        assert len(positions.shape) <= 2

        if len(positions.shape) == 1:
            positions = positions.reshape(1, 3)

        return _batch_query(self._scale, self._node_index, self._leaf_index,
                            positions)

    def intersect(self, starts: np.ndarray, directions: np.ndarray,
                  max_length: int) -> Path:
        """Intersects a ray with the tree.

        Args:
            starts (np.ndarray): the starting points of the rays
            directions (np.ndarray): the directions of the rays

        Returns:
            List[Voxel]: A list of (t, id) leaf voxels visited.
        """
        assert starts.shape[-1] == 3
        assert directions.shape[-1] == 3
        assert len(starts.shape) <= 2
        assert len(directions.shape) <= 2
        assert len(starts.shape) == len(directions.shape)

        if len(starts.shape) == 1:
            starts = starts.reshape(1, 3)
            directions = directions.reshape(1, 3)

        directions = np.where(directions == 0, 1e-8, directions)

        return _batch_intersect(self._scale, self._node_index, self._leaf_index,
                                starts, directions, max_length)

    @staticmethod
    def build_from_samples(positions: np.ndarray,
                           depth: int,
                           min_leaf_size: int,
                           data: np.ndarray = None) -> "OcTree":
        """Builds a sparse OcTree from the provided position samples.

        Args:
            positions (np.ndarray): a (N,3) tensor of positions
            data (np.ndarray): a (N,3) tensor of position data
            depth (int): The depth at which stop growing the tree.
            min_leaf_size (int): The minimum number of contained positions
                                 needed for a leaf to be created.

        Returns:
            OcTree: the constructed sparse OcTree
            leaf_data: the mean of the data contained in a leaf
        """
        if data is None:
            data = np.zeros((len(positions), 1))
            no_data = True
        else:
            no_data = False

        min_pos = positions.min(0)
        max_pos = positions.max(0)
        scale = (max_pos - min_pos).max() * 0.5
        center = 0.5 * (min_pos + max_pos)
        positions -= center
        node = Node(0, 0, 0, 0, scale, 0)
        index = np.arange(len(positions))
        queue = deque([(node, index)])
        node_ids = set()
        leaf_ids = set()
        leaf_data = []
        bar = ChargingBar("Generating voxels", max=len(positions))
        while queue:
            node, index = queue.popleft()
            if node.depth == depth - 1:
                # at target depth, let's see if we can create a leaf.
                bar.next(len(index))
                if len(index) >= min_leaf_size:
                    leaf_ids.add(node.id)
                    leaf_data.append(data[index].mean(0))
            elif node.depth < depth - 1:
                # need to keep splitting at the hyperplanes
                node_ids.add(node.id)
                assignment = _batch_assign(node, positions[index])
                valid_child = False
                invalid_length = 0
                children = _list_children(node)
                for i, child in enumerate(children):
                    child_index = index[assignment == i]
                    if len(child_index) >= min_leaf_size:
                        # potential child, add it to the search
                        queue.append((child, child_index))
                        valid_child = True
                    else:
                        invalid_length += len(child_index)

                if valid_child:
                    # there was at least one valid child, so we
                    # just discard the remaining positions
                    bar.next(invalid_length)
                else:
                    # no valid children, which makes this is a leaf
                    bar.next(len(index))
                    leaf_ids.add(node.id)
                    leaf_data.append(data[index].mean(0))

        bar.finish()
        leaf_data = None if no_data else np.stack(leaf_data)
        return OcTree(scale, node_ids, leaf_ids, leaf_data)

    @staticmethod
    def build_from_mesh(mesh_path: str, voxel_depth: int,
                        min_leaf_size: int, up_dir=(0, 1, 0)) -> "OcTree":
        """Builds an OcTree from a mesh.

        Description:
            Builds an octree by performing barycentric sampling on the surface
            of a mesh to create a dense point cloud, and then fitting the
            OcTree to the resulting cloud to the specified depth.

        Args:
            mesh_path (str): the path to the mesh file. Must be understandable
                             by `trimesh`. The mesh in question is assumed
                             to be a UV mesh with an associated texture, which
                             is used to gather color data to store in the
                             leaves.
            voxel_depth (int): Maximum depth of the tree
            min_leaf_size (int): The minimum number of points required to create
                                 a leaf.
            up_dir (Tuple[int], optional): The "up direction" to use when
                                           normalizing the points. The resulting
                                           octree will be oriented such that
                                           this direction is y+.

        Returns:
            OcTree: the constructed tree.
        """
        up_dir = np.array(up_dir, np.float32)

        print("Loading model...")
        mesh = trimesh.load(mesh_path)
        mesh.vertices = _normalize_points(mesh.vertices, up_dir)
        verts = np.array(mesh.vertices, np.float32)
        triangles = np.array(mesh.faces, np.int64)
        uvs = np.array(mesh.visual.uv, np.float32)
        num_positions = (8 ** (voxel_depth - 2)) * min_leaf_size

        print("Sampling", num_positions, "positions on the surface of the mesh")
        verts, uvs = _sample_barycentric_point_cloud(verts, triangles, uvs,
                                                     num_positions)
        texture = np.array(mesh.visual.material.image)[::-1]
        colors = interpolate_bilinear(texture, uvs)[..., :3]
        colors = (colors / 255).astype(np.float32)

        print("Building the octree")
        return OcTree.build_from_samples(verts, voxel_depth,
                                         min_leaf_size, colors)

    def _update(self, node_ids: Set[int], leaf_ids: Set[int], scale: float):
        self._scale = scale
        self._leaf_ids = leaf_ids
        self._node_ids = node_ids - leaf_ids
        node_index = list(sorted(self._node_ids))
        self._node_index = np.array(node_index)
        leaf_index = list(sorted(self._leaf_ids))
        self._leaf_index = np.array(leaf_index)
        if len(self._node_ids):
            self.leaves = _leaf_nodes(self._scale, self._node_ids, self._leaf_ids)
        else:
            self.leaves = [Node(0, 0, 0, 0, 1, 0)]

    @property
    def state_dict(self) -> Dict[str, np.ndarray]:
        """The state needed to reconstruct the OcTree."""
        state_dict = {
            "node_index": self._node_index,
            "leaf_index": self._leaf_index,
            "scale": self._scale
        }

        if self._leaf_data is not None:
            state_dict["leaf_data"] = self._leaf_data

        return state_dict

    def save(self, path: str):
        """Saves the OcTree to the provided path."""
        state = self.state_dict
        np.savez(path, **state)

    @staticmethod
    def load(path_or_data: Union[str, Dict[str, np.ndarray]]) -> "OcTree":
        """Loads the OcTree.

        Args:
            path_or_data: either the path to the data file, or a state dict.

        Returns:
            The stored OcTree
        """
        if isinstance(path_or_data, str):
            if not os.path.exists(path_or_data):
                path = os.path.join(os.path.dirname(__file__), "..", "data", path_or_data)
                path = os.path.abspath(path)
                if not os.path.exists(path):
                    print("Downloading octree...")
                    dataset_name = os.path.basename(path)
                    success = download_asset(dataset_name, path)
                    if not success:
                        print("Unable to download octree", dataset_name)
                        return None
            else:
                path = path_or_data

            data = np.load(path)
        else:
            data = path_or_data

        scale = float(data["scale"])
        node_ids = set([int(index) for index in data["node_index"]])
        leaf_ids = set([int(index) for index in data["leaf_index"]])
        leaf_data = data["leaf_data"] if "leaf_data" in data else None

        return OcTree(scale, node_ids, leaf_ids, leaf_data)

    def load_state(self, state_dict: Dict[str, np.ndarray]):
        """Loads the information from the state dictionary."""
        node_ids = set([int(index) for index in state_dict["node_index"]])
        leaf_ids = set([int(index) for index in state_dict["leaf_index"]])
        scale = float(state_dict["scale"])
        self._update(node_ids, leaf_ids, scale)
