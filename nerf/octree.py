"""Attempt at coding an octree using Numba."""

from collections import deque, namedtuple
from itertools import product
from typing import List, Sequence, Set

from numba import njit, prange
import numpy as np


Vector = namedtuple("Vector", ["x", "y", "z"])
Node = namedtuple("Node", ["id", "x", "y", "z", "scale"])
Ray = namedtuple("Ray", ["x", "y", "z", "dx", "dy", "dz"])
Intersection = namedtuple("Intersection", ["t_min", "t_max"])
Path = namedtuple("Path", ["t_stops", "leaves"])


@njit
def _intersect_node_with_ray(node: Node, ray: Ray) -> Intersection:
    dx = node.x - ray.x
    dy = node.y - ray.y
    dz = node.z - ray.z
    x0 = (dx - node.scale) / ray.dx
    y0 = (dy - node.scale) / ray.dy
    z0 = (dz - node.scale) / ray.dz
    x1 = (dx + node.scale) / ray.dx
    y1 = (dy + node.scale) / ray.dy
    z1 = (dz + node.scale) / ray.dz

    if x1 < x0:
        tmp = x0
        x0 = x1
        x1 = tmp

    if y1 < y0:
        tmp = y0
        y0 = y1
        y1 = tmp

    if z1 < z0:
        tmp = z0
        z0 = z1
        z1 = tmp

    return Intersection(max(x0, y0, z0), min(x1, y1, z1))


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
                return Node(child_id, x, y, z, scale)
            else:
                z = node.z + scale
                child_id += 1
                return Node(child_id, x, y, z, scale)
        else:
            y = node.y + scale
            child_id += 2
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale)
            else:
                z = node.z + scale
                child_id += 1
                return Node(child_id, x, y, z, scale)
    else:
        x = node.x + scale
        child_id += 4
        if point.y < node.y:
            y = node.y - scale
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale)
            else:
                z = node.z + scale
                child_id += 1
                return Node(child_id, x, y, z, scale)
        else:
            y = node.y + scale
            child_id += 2
            if point.z < node.z:
                z = node.z - scale
                return Node(child_id, x, y, z, scale)
            else:
                z = node.z + scale
                child_id += 1
                return Node(child_id, x, y, z, scale)


@njit
def _trace_ray_path(scale: float, leaf_index: np.ndarray,
                    start: np.ndarray, direction: np.ndarray,
                    t_stops: np.ndarray, leaves: np.ndarray) -> Path:
    stack = [Node(0, 0.0, 0.0, 0.0, scale)]
    max_length = len(t_stops)
    ray = Ray(start[0], start[1], start[2],
              direction[0], direction[1], direction[2])
    t, t_max = _intersect_node_with_ray(stack[0], ray)
    t += 1e-5
    p = _cast_ray(ray, t)
    stop = 0
    while stack:
        current = stack[-1]

        index = np.searchsorted(leaf_index, current.id)
        if leaf_index[index] == current.id:
            _, tc_max = _intersect_node_with_ray(current, ray)
            t_stops[stop] = t
            leaves[stop] = index
            t = tc_max + 1e-5
            p = _cast_ray(ray, t)
            while _node_contains(current, p):
                t += 1e-5
                p = _cast_ray(ray, t)

            stack.pop()
            stop += 1
            if t == t_max or stop == max_length:
                break
        else:
            if _node_contains(current, p):
                child = _find_child_of_node(current, p)
                stack.append(child)
            else:
                stack.pop()


@njit(parallel=True)
def _batch_intersect(scale: float,
                     leaf_index: np.ndarray,
                     starts: np.ndarray,
                     directions: np.ndarray,
                     max_length: int) -> Path:
    num_rays = starts.shape[0]
    t_stops = np.zeros((num_rays, max_length), np.float32)
    leaves = np.zeros((num_rays, max_length), np.int64)
    for i in prange(num_rays):
        _trace_ray_path(scale, leaf_index, starts[i], directions[i],
                        t_stops[i], leaves[i])

    return Path(t_stops, leaves)


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
        yield Node(child_id + i, x, y, z, scale)


def _leaf_nodes(scale: float, leaf_ids: Set[int]) -> List[Node]:
    queue = deque([Node(0, 0.0, 0.0, 0.0, scale)])
    leaves = []
    while queue:
        current = queue.popleft()
        if current.id in leaf_ids:
            leaves.append(current)
        else:
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

        self._leaf_ids = set()
        self._branch_ids = set()
        queue = deque([(0, 0)])
        while queue:
            current, level = queue.popleft()
            if level == depth - 2:
                self._branch_ids.add(current)
            elif level == depth - 1:
                self._leaf_ids.add(current)
                continue

            child_start = (current << 3) + 1
            child_end = child_start + 8
            for child_id in range(child_start, child_end):
                queue.append((child_id, level + 1))

    def leaf_centers(self) -> np.ndarray:
        """The Nx3 center coordinates of all leaves."""
        leaves = _leaf_nodes(self._scale, self._leaf_ids)
        leaf_centers = [[leaf.x, leaf.y, leaf.z] for leaf in leaves]
        return np.array(leaf_centers, np.float32)

    def leaf_scales(self) -> np.ndarray:
        """The N scale values for all leaves."""
        leaves = _leaf_nodes(self._scale, self._leaf_ids)
        leaf_scales = [leaf.scale for leaf in leaves]
        return np.array(leaf_scales, np.float32)

    def _split(self, leaf_id: int):
        self._leaf_ids.remove(leaf_id)
        self._branch_ids.add(leaf_id)
        parent_id = (leaf_id - 1) >> 3
        if parent_id in self._branch_ids:
            self._branch_ids.remove(parent_id)

        leaf_start = (leaf_id << 3) + 1
        leaf_end = leaf_start + 8
        for i in range(leaf_start, leaf_end):
            self._leaf_ids.add(i)

    def _merge(self, branch_id: int):
        self._branch_ids.remove(branch_id)
        self._leaf_ids.add(branch_id)
        leaf_start = (branch_id << 3) + 1
        leaf_end = leaf_start + 8
        for i in range(leaf_start, leaf_end):
            self._leaf_ids.remove(i)

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

    def merge(self, leaf_values: np.ndarray, threshold: float) -> np.ndarray:
        """Merges nodes using the values provided.

        Description:
            Branch nodes (i.e. nodes with only leaves as children) will be
            merged if they contain values which are mostly "on" (i.e. equal to
            1) or mostly "off" (i.e. equal to zero). This is determined by
            measuring the mean value and comparing it to the provided threshold.

        Args:
            leaf_values (np.ndarray): values used to determine which branches
                                      should be merged (0-1).
            threshold (float): the threshold value used to select merge
                               candidates.

        Returns:
            np.ndarray: the updated leaf values after merging.
        """
        leaf_opacity = {}
        branch_opacity = {}

        for leaf_id, value in zip(sorted(self._leaf_ids), leaf_values):
            parent_id = (leaf_id - 1) >> 3
            leaf_opacity[leaf_id] = value
            if parent_id not in branch_opacity:
                branch_opacity[parent_id] = 0

            branch_opacity[parent_id] += value / 8

        # sort branches by uncertainty (least to most certain)
        branch_ids = list(sorted(self._branch_ids))
        branch_values = np.array([branch_opacity[i] for i in branch_ids])
        branch_values = np.abs(branch_values - 0.5)
        sorted_branch_ids = np.argsort(branch_values)
        sorted_branch_ids = [branch_ids[idx] for idx in reversed(sorted_branch_ids)]

        # determine how many merges we can make while
        num_merge = (branch_values > 0.5 - threshold).sum()

        print("Performing", num_merge, "merges")
        for branch_id in sorted_branch_ids[:num_merge]:
            self._merge(branch_id)

        print("Num nodes:", len(self))

        # compute the updated opacity values
        opacity = []
        for leaf_id in sorted(self._leaf_ids):
            if leaf_id in leaf_opacity:
                opacity.append(leaf_opacity[leaf_id])
            elif leaf_id in branch_opacity:
                opacity.append(branch_opacity[leaf_id])
            else:
                raise KeyError()

        # this is necessary as some nodes may have had
        # all of their branch children merged
        self._branch_ids.clear()
        queue = deque([0])
        while queue:
            current = queue.popleft()
            child_start = (current << 3) + 1
            child_end = child_start + 8
            is_branch = True
            for child_id in range(child_start, child_end):
                if child_id not in self._leaf_ids:
                    is_branch = False
                    queue.append(child_id)

            if is_branch:
                self._branch_ids.add(current)

        return np.array(opacity, np.float32)

    def split(self, leaf_values: np.ndarray, threshold: float, max_nodes: int) -> np.ndarray:
        """Splits nodes using the values provided.

        Description:
            This method will determine which nodes are most uncertain to be
            opaque, that is having a value close to 0.5, and then splits them
            until the maximum number of nodes is reached.

        Args:
            leaf_values: the opacity values used for splitting the leaves (0-1)
            threshold: the threshold used to determine split candidates
            max_nodes: maximum number of nodes in the tree

        Returns:
            np.ndarray: the updated leaf values after splitting
        """
        leaf_opacity = {}

        leaf_ids = list(sorted(self._leaf_ids))
        for leaf_id, value in zip(leaf_ids, leaf_values):
            leaf_opacity[leaf_id] = value

        # sort branches by uncertainty (least to most certain)
        leaf_values = np.abs(leaf_values - 0.5)
        sorted_leaf_ids = np.argsort(leaf_values)
        sorted_leaf_ids = [leaf_ids[idx] for idx in sorted_leaf_ids]

        # determine how many splits we can make
        num_split = (leaf_values < threshold).sum()
        budget = (max_nodes - len(self)) // 8
        num_split = min(num_split, budget)

        print("Performing", num_split, "splits")
        for leaf_id in sorted_leaf_ids[:num_split]:
            self._split(leaf_id)

        print("Num nodes:", len(self))

        # compute the updated opacity values
        opacity = []
        for leaf_id in sorted(self._leaf_ids):
            parent_id = (leaf_id - 1) >> 3
            if leaf_id in leaf_opacity:
                opacity.append(leaf_opacity[leaf_id])
            elif parent_id in leaf_opacity:
                opacity.append(leaf_opacity[parent_id])
            else:
                raise KeyError()

        return np.array(opacity, np.float32)

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

        leaf_index = list(sorted(self._leaf_ids))
        leaf_index = np.array(leaf_index)
        path = _batch_intersect(self._scale, leaf_index, starts, directions, max_length)
        _batch_intersect.parallel_diagnostics(level=4)
        return path
