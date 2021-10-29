"""Attempt at coding an octree using Numba."""

from collections import deque
from itertools import product
from typing import Dict, List, NamedTuple, Sequence, Set, Tuple

from numba import njit
import numpy as np
from progress.bar import ChargingBar


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

        index = np.searchsorted(leaf_index, current.id)
        if leaf_index[index] == current.id:
            tc = _intersect_node_with_ray(current, ray)
            t_stops[stop] = t
            leaves[stop] = index

            stack.pop()
            stop += 1
            if t >= tr.t_max or stop == max_length - 1:
                break

            t = tc.t_max + 1e-5
            point = _cast_ray(ray, t)
            while _node_contains(current, point):
                t += 1e-5
                point = _cast_ray(ray, t)

            sibling = _find_sibling_of_node(current, point, tc.a_max)
            if sibling != current:
                stack.append(sibling)
        else:
            index = np.searchsorted(node_index, current.id)
            if node_index[index] == current.id:
                if _node_contains(current, point):
                    child = _find_child_of_node(current, point)
                    stack.append(child)
                else:
                    stack.pop()
            else:
                stack.pop()

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


def _enumerate_children(node: Node) -> Sequence[Node]:
    scale = node.scale / 2
    child_id = (node.id << 3) + 1
    x0 = node.x - scale
    y0 = node.y - scale
    z0 = node.z - scale
    x1 = node.x + scale
    y1 = node.y + scale
    z1 = node.z + scale
    for i, (x, y, z) in enumerate(product([x0, x1], [y0, y1], [z0, z1])):
        yield Node(child_id + i, x, y, z, scale, node.depth + 1)


def _leaf_nodes(scale: float, node_ids: Set[int], leaf_ids: Set[int]) -> List[Node]:
    queue = deque([Node(0, 0.0, 0.0, 0.0, scale, 0)])
    leaves = []
    while queue:
        current = queue.popleft()
        if current.id in leaf_ids:
            leaves.append(current)
        elif current.id in node_ids:
            for child in _enumerate_children(current):
                queue.append(child)

    return leaves


class OcTree:
    """Class representing an OcTree datastructure."""

    def __init__(self, depth: int, scale=1.0):
        """Constructor.

        Args:
            depth (int): the initial depth of the tree.
            scale (float): the scale of the octree, equal to one-half of a side.
                           Defaults to 1.0.
        """
        self._scale = float(scale)

        self._node_ids = set()
        self._leaf_ids = set()
        queue = deque([(0, 0)])
        while queue:
            current, level = queue.popleft()
            if level == depth - 1:
                self._leaf_ids.add(current)
                continue

            self._node_ids.add(current)
            child_start = (current << 3) + 1
            child_end = child_start + 8
            for child_id in range(child_start, child_end):
                queue.append((child_id, level + 1))

        leaf_index = list(sorted(self._leaf_ids))
        self._leaf_index = np.array(leaf_index)
        node_index = list(sorted(self._node_ids))
        self._node_index = np.array(node_index)

    def leaf_centers(self) -> np.ndarray:
        """The Nx3 center coordinates of all leaves."""
        leaves = _leaf_nodes(self._scale, self._node_ids, self._leaf_ids)
        leaf_centers = [[leaf.x, leaf.y, leaf.z] for leaf in leaves]
        return np.array(leaf_centers, np.float32)

    def leaf_depths(self) -> np.ndarray:
        """The N depths for all leaves."""
        leaves = _leaf_nodes(self._scale, self._node_ids, self._leaf_ids)
        leaf_depths = [leaf.depth for leaf in leaves]
        return np.array(leaf_depths, np.int32)

    def __len__(self):
        """Counts all the nodes in the tree."""
        num_nodes = 0
        stack = [0]
        while stack:
            current = stack.pop()
            num_nodes += 1
            if current in self._leaf_ids:
                continue

            child_start = (current << 3) + 1
            child_end = child_start + 8
            for child_id in range(child_start, child_end):
                stack.append(child_id)

        return num_nodes

    @property
    def num_leaves(self) -> int:
        """Counts the number of leaves in the tree."""
        return len(self._leaf_ids)

    @property
    def scale(self) -> float:
        """Scale of the cube (side is 2 * scale)."""
        return self._scale

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
                           colors: np.ndarray,
                           depth: int,
                           min_leaf_size: int) -> Tuple["OcTree", np.ndarray]:
        min_pos = positions.min(0)
        max_pos = positions.max(0)
        scale = (max_pos - min_pos).max() * 0.5
        node = Node(0, 0, 0, 0, scale, 0)
        index = np.arange(len(positions))
        queue = deque([(node, index)])
        node_ids = set()
        leaf_ids = set()
        leaf_colors = []
        bar = ChargingBar("Generating voxels", max=len(positions))
        while queue:
            node, index = queue.popleft()
            if node.depth == depth:
                bar.next(len(index))
                if len(index) >= min_leaf_size:
                    color = colors[index].mean(0)
                    leaf_ids.add(node.id)
                    leaf_colors.append(color)
            elif node.depth < depth:
                node_ids.add(node.id)
                assignment = _batch_assign(node, positions[index])
                for i, child in enumerate(_enumerate_children(node)):
                    child_index = index[assignment == i]
                    if len(child_index):
                        queue.append((child, child_index))

        bar.finish()
        leaf_colors = np.stack(leaf_colors)
        result = OcTree(1)
        result._leaf_ids = leaf_ids
        result._node_ids = node_ids - result._leaf_ids
        result._scale = scale
        return result, leaf_colors

    @property
    def state_dict(self):
        leaf_index = list(sorted(self._leaf_ids))
        leaf_index = np.array(leaf_index, np.int64)
        return {
            "leaf_index": leaf_index,
            "scale": self._scale
        }

    def save(self, path: str):
        state = self.state_dict
        np.savez(path, **state)

    @staticmethod
    def load(path_or_data) -> "OcTree":
        if isinstance(path_or_data, str):
            data = np.load(path_or_data)
        else:
            data = path_or_data

        result = OcTree(1)
        result.load_state(data)
        return result

    def load_state(self, state_dict: Dict[str, np.ndarray]):
        self._leaf_ids = set([int(index) for index in state_dict["leaf_index"]])
        self._scale = float(state_dict["scale"])
