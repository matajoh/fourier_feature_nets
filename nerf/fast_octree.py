"""Attempt at coding an octree using Numba."""

from collections import deque, namedtuple
from itertools import product
from typing import Dict, List, Sequence, Set, Tuple

from numba import njit
import numpy as np


Node = namedtuple("Node", ["id", "x", "y", "z", "scale"])
Path = namedtuple("Path", ["t_stops", "leaves"])


@njit
def _intersect_near(centers: np.ndarray,
                    scales: np.ndarray,
                    starts: np.ndarray,
                    dirs_inv: np.ndarray) -> np.ndarray:
    diffs = centers - starts
    min_bound = (diffs - scales) * dirs_inv
    max_bound = (diffs + scales) * dirs_inv

    tc_min = np.where(min_bound < max_bound, min_bound, max_bound)
    tc_x = tc_min[:, 0]
    tc_y = tc_min[:, 1]
    tc_z = tc_min[:, 2]
    tc = np.where(tc_x > tc_y, tc_x, tc_y)
    tc = np.where(tc > tc_z, tc, tc_z)

    return tc


@njit
def _intersect_far(centers: np.ndarray,
                   scales: np.ndarray,
                   starts: np.ndarray,
                   dirs_inv: np.ndarray) -> np.ndarray:
    diffs = centers - starts
    min_bound = (diffs - scales) * dirs_inv
    max_bound = (diffs + scales) * dirs_inv

    tc_max = np.where(min_bound > max_bound, min_bound, max_bound)
    tc_x = tc_max[:, 0]
    tc_y = tc_max[:, 1]
    tc_z = tc_max[:, 2]
    tc = np.where(tc_x < tc_y, tc_x, tc_y)
    tc = np.where(tc < tc_z, tc, tc_z)

    return tc


@njit
def _contains(centers: np.ndarray,
              scales: np.ndarray,
              points: np.ndarray) -> np.ndarray:
    diff = np.abs(points - centers) > scales
    diff_x = diff[:, 0]
    diff_y = diff[:, 1]
    diff_z = diff[:, 2]
    diff = diff_x | diff_y | diff_z
    return np.logical_not(diff)


@njit
def _find_child(centers: np.ndarray,
                scales: np.ndarray,
                node_ids: np.ndarray,
                points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    signs = np.sign(points - centers)
    child_centers = centers + (signs * scales * 0.5).astype(np.float32)
    bits = signs > 0
    starts = (node_ids << 3) + 1
    child_ids = (bits[:, 0] << 2) + (bits[:, 1] << 1) + bits[:, 2] + starts
    return child_ids, child_centers


@njit
def _find_parent(centers: np.ndarray,
                 scales: np.ndarray,
                 node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    parent_ids = (node_ids - 1) >> 3
    starts = (parent_ids << 3) + 1
    child_ids = (node_ids - starts).astype(np.uint8)
    bits = np.zeros((len(starts), 3), np.uint8)
    bits[:, 0] = (child_ids & 4) >> 2
    bits[:, 1] = (child_ids & 2) >> 1
    bits[:, 2] = (child_ids & 1)
    signs = np.where(bits, 1, -1).astype(np.float32)
    parent_centers = centers - (signs * scales)
    return parent_ids, parent_centers


@njit
def _find_sibling(centers: np.ndarray,
                  scales: np.ndarray,
                  node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass


@njit
def _batch_intersect(scale: float,
                     leaf_index: np.ndarray,
                     starts: np.ndarray,
                     directions: np.ndarray,
                     max_length: int,
                     epsilon=1e-8) -> Path:

    directions = np.where(np.abs(directions) < epsilon,
                          np.full_like(directions, epsilon),
                          directions)
    dirs_inv = np.reciprocal(directions)

    num_rays = len(starts)
    current_node_ids = np.zeros(num_rays, np.int64)
    current_index = np.zeros(num_rays, np.int64)
    current_centers = np.zeros((num_rays, 3), np.float32)
    current_scales = np.full(num_rays, scale, np.float32)
    t_stops = np.zeros((num_rays, max_length), np.float32)
    leaves = np.zeros((num_rays, max_length), np.int64)
    t = _intersect_near(current_centers, current_scales.reshape(-1, 1), starts, dirs_inv)
    t_max = _intersect_far(current_centers, current_scales.reshape(-1, 1), starts, dirs_inv)
    t += 1e-5
    in_progress = current_index < max_length - 1
    points = starts + t.reshape(-1, 1) * directions
    while in_progress.any():
        current_leaves = np.searchsorted(leaf_index, current_node_ids)
        is_leaf = leaf_index[current_leaves] == current_node_ids
        not_leaf = np.logical_not(is_leaf)
        contains = _contains(current_centers, current_scales.reshape(-1, 1), points)

        tc_max = _intersect_far(current_centers[is_leaf],
                                current_scales[is_leaf].reshape(-1, 1),
                                starts[is_leaf],
                                dirs_inv[is_leaf])

        tc_max += 1e-5
        ray_idx = np.nonzero(is_leaf)[0]
        for ray, tc in zip(ray_idx, tc_max):
            idx = current_index[ray]
            t_stops[ray, idx] = t[ray]
            leaves[ray, idx] = current_leaves[ray]
            t[ray] = tc + 1e-5

        current_index = np.where(is_leaf & in_progress, current_index + 1, current_index)
        points = starts + t.reshape(-1, 1) * directions

        in_progress = (current_index == max_length - 1) | (t >= t_max) | (current_node_ids == -1)
        in_progress = np.logical_not(in_progress)
        push = contains & not_leaf
        pop = np.logical_not(push)
        push = push & in_progress
        pop = pop & in_progress

        child_ids, child_centers = _find_child(current_centers,
                                               current_scales.reshape(-1, 1),
                                               current_node_ids,
                                               points)

        parent_ids, parent_centers = _find_parent(current_centers,
                                                  current_scales.reshape(-1, 1),
                                                  current_node_ids[:])

        current_node_ids = np.where(push, child_ids, current_node_ids)
        current_node_ids = np.where(pop, parent_ids, current_node_ids)
        current_scales = np.where(push, current_scales / 2, current_scales)
        current_scales = np.where(pop, current_scales * 2, current_scales)
        push = np.repeat(push, 3).reshape(-1, 3)
        pop = np.repeat(pop, 3).reshape(-1, 3)
        current_centers = np.where(push, child_centers, current_centers)
        current_centers = np.where(pop, parent_centers, current_centers)

    for i, end in enumerate(current_index):
        t_stops[i, end:] = t_max[i]
        leaves[i, end:] = -1

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


class FastOcTree:
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
        self._update_branches()

        return np.array(opacity, np.float32)

    def _update_branches(self):
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

        starts = starts.astype(np.float32)
        directions = directions.astype(np.float32)
        leaf_index = list(sorted(self._leaf_ids))
        leaf_index = np.array(leaf_index, np.int64)
        path = _batch_intersect(self._scale, leaf_index, starts, directions, max_length)
        return path

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
    def load(path_or_data) -> "FastOcTree":
        if isinstance(path_or_data, str):
            data = np.load(path_or_data)
        else:
            data = path_or_data

        result = FastOcTree(1)
        result.load_state(data)
        return result

    def load_state(self, state_dict: Dict[str, np.ndarray]):
        self._leaf_ids = set(state_dict["leaf_index"].tolist())
        self._scale = state_dict["scale"]
        self._update_branches()
