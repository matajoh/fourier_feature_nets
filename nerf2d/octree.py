from collections import namedtuple
from itertools import product
from typing import Set, Sequence, List, Tuple

import numpy as np

CORNERS = np.array(list(product([-1, 1], [-1, 1], [-1, 1])), np.float32)

class OcTree:
    class Node(namedtuple("Node", ["id", "scale", "center", "parent"])):
        @property
        def child_ids(self) -> Sequence[int]:           
            start = (self.id << 3) + 1
            return range(start, start + 8)

        def make_children(self) -> Sequence["OcTree.Node"]:
            scale = self.scale / 2
            offset = scale * CORNERS
            for child, center in zip(self.child_ids, offset):
                yield OcTree.Node(child, scale, self.center + center, self.id)

        def intersect_ray(self, point: np.ndarray, dir_inv: np.ndarray):
            bounds = np.stack([self.center - self.scale, self.center + self.scale])
            with np.errstate(invalid='ignore'):
                t = np.nan_to_num((bounds - point) * dir_inv)

            tc_min = t.min(0).max()
            tc_max = t.max(0).min()
            return tc_min, tc_max

        def contains(self, point: np.ndarray) -> bool:
            diff = np.abs(point - self.center) <= self.scale
            return diff.all()

        def find_child(self, point: np.ndarray):
            bits = point >= self.center
            start = (self.id << 3) + 1
            return start + np.packbits(bits[::-1], bitorder="little")[0]

    def __init__(self, initial_depth: int, max_nodes: int):
        self.max_nodes = max_nodes
        self.nodes = {
            0: OcTree.Node(0, 1, np.zeros(3, np.float32), None)
        }

        self.leaves: Set[OcTree.Node] = set()
        self.branches: Set[OcTree.Node] = set()

        stack = [(self.nodes[0], 1)]
        
        while stack:
            current, depth = stack.pop()
            for child in current.make_children():
                self.nodes[child.id] = child
                if depth == initial_depth - 1:
                    self.leaves.add(child.id)
                else:
                    stack.append((child, depth + 1))
                    if depth == initial_depth - 2:
                        self.branches.add(child.id)

    def leaf_centers(self):
        return np.stack([self.nodes[i].center for i in sorted(self.leaves)])

    def leaf_scales(self):
        return np.stack([self.nodes[i].scale for i in sorted(self.leaves)])

    def _split(self, leaf_id: int):
        current = self.nodes[leaf_id]
        for child in current.make_children():
            self.nodes[child.id] = child
            self.leaves.add(child.id)

        self.leaves.remove(leaf_id)
        self.branches.add(leaf_id)

    def _merge(self, branch_id: int):
        current = self.nodes[branch_id]
        for child_id in current.child_ids:
            del self.nodes[child_id]
            self.leaves.remove(child_id)

        self.leaves.add(branch_id)
        self.branches.remove(branch_id)

    def merge(self, leaf_values: np.ndarray, threshold: float) -> np.ndarray:
        leaf_opacity = {}
        branch_opacity = {}

        for leaf_id, value in zip(sorted(self.leaves), leaf_values):
            leaf = self.nodes[leaf_id]
            leaf_opacity[leaf_id] = value
            if not leaf.parent in branch_opacity:
                branch_opacity[leaf.parent] = 0
            
            branch_opacity[leaf.parent] += value / 8

        # sort branches by uncertainty (least to most certain)
        branch_ids = list(sorted(self.branches))
        branch_values = np.array([branch_opacity[i] for i in branch_ids])
        branch_values = np.abs(branch_values - 0.5)
        sorted_branch_ids = np.argsort(branch_values)
        sorted_branch_ids = [branch_ids[idx] for idx in reversed(sorted_branch_ids)]

        # determine how many merges we can make while
        num_merge = (branch_values > 0.5 - threshold).sum()

        print("Performing", num_merge, "merges")
        for branch_id in sorted_branch_ids[:num_merge]:
            self._merge(branch_id)

        print("Num nodes:", len(self.nodes))

        # compute the updated opacity values
        opacity = []
        for leaf_id in sorted(self.leaves):
            if leaf_id in leaf_opacity:
                opacity.append(leaf_opacity[leaf_id])
            elif leaf_id in branch_opacity:
                opacity.append(branch_opacity[leaf_id])
            else:
                raise KeyError()

        self.branches.clear()
        for node in self.nodes.values():
            if node.id in self.leaves:
                continue

            is_branch = True
            for child_id in node.child_ids:
                if child_id not in self.leaves:
                    is_branch = False
                    break
            
            if is_branch:
                self.branches.add(node.id)

        return np.array(opacity, np.float32)

    def split(self, leaf_values: np.ndarray, threshold: float) -> np.ndarray:
        leaf_opacity = {}

        leaf_ids = list(sorted(self.leaves))
        for leaf_id, value in zip(leaf_ids, leaf_values):
            leaf_opacity[leaf_id] = value

        # sort branches by uncertainty (least to most certain)
        leaf_values = np.abs(leaf_values - 0.5)
        sorted_leaf_ids = np.argsort(leaf_values)
        sorted_leaf_ids = [leaf_ids[idx] for idx in sorted_leaf_ids]

        # determine how many merges we can make while
        num_split = (leaf_values < threshold).sum()
        budget = (self.max_nodes - len(self.nodes)) // 8
        num_split = min(num_split, budget)

        print("Performing", num_split, "splits")
        for leaf_id in sorted_leaf_ids[:num_split]:
            self._split(leaf_id)

        print("Num nodes:", len(self.nodes))

        # compute the updated opacity values
        opacity = []
        for leaf_id in sorted(self.leaves):
            leaf = self.nodes[leaf_id]
            if leaf_id in leaf_opacity:
                opacity.append(leaf_opacity[leaf_id])
            elif leaf.parent in leaf_opacity:
                opacity.append(leaf_opacity[leaf.parent])
            else:
                raise KeyError()

        return np.array(opacity, np.float32)

    def save(self, path: str):
        ids = []
        scales = []
        centers = []
        parents = []
        for node in self.nodes.values():
            ids.append(node.id)
            scales.append(node.scale)
            centers.append(node.center)
            parents.append(node.parent)

        np.savez(path,
                 num_nodes=len(self.nodes),
                 ids=ids,
                 scales=scales,
                 centers=centers,
                 parents=parents)

    def load(self, path: str):
        data = np.load(path)
        result = OcTree(0, data["num_nodes"])
        ids = data["ids"]
        scales = data["scales"]
        centers = data["centers"]
        parents = data["parents"]
        for id, scale, center, parent in zip(ids, scales, centers, parents):
            result.nodes[id] = OcTree.Node(id, scale, center, parent)

    @staticmethod
    def _batch_intersect(bounds: np.ndarray, points: np.ndarray, dirs_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid='ignore'):
            t = np.nan_to_num((bounds - points[:, np.newaxis, :]) * dirs_inv[:, np.newaxis, :])

        tc_min = t.min(1).max(-1)
        tc_max = t.max(1).min(-1)
        return tc_min, tc_max

    @staticmethod
    def _batch_contains(centers: np.ndarray, scales: np.ndarray, points: np.ndarray) -> np.ndarray:
        diff = np.abs(points - centers) <= scales
        return diff.all(-1)

    @staticmethod
    def _batch_find_child(centers: np.ndarray, node_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
        bits = points >= centers
        starts = (node_ids << 3) + 1
        return starts + np.packbits(bits[:, ::-1], axis=-1, bitorder="little").reshape(-1)

    def batch_intersect(self, points: np.ndarray, directions: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            dirs_inv = np.reciprocal(directions)

        node_index = np.array(list(self.nodes.keys()), np.int64)
        node_index.sort()

        node_leaves = np.array(list(self.leaves), np.int64)
        node_leaves.sort()

        node_centers = [self.nodes[i].center for i in node_index]
        node_centers = np.stack(node_centers)

        node_scales = [self.nodes[i].scale for i in node_index]
        node_scales = np.stack(node_scales).reshape(-1, 1)

        node_bounds = np.stack([node_centers - node_scales, node_centers + node_scales], 1)

        current_node_ids = np.zeros(len(points), np.int64)
        t_stops = []
        node_stops = []
        leaf_stops = []
        t, t_max = self._batch_intersect(node_bounds[current_node_ids], points, dirs_inv)
        t += 1e-5
        while (t < t_max).any():
            t_stops.append(t.copy())
            node_stops.append(current_node_ids.copy())

            current_index = node_index.searchsorted(current_node_ids)
            centers = node_centers[current_index]
            scales = node_scales[current_index]
            bounds = node_bounds[current_index]

            is_leaf = node_leaves.searchsorted(current_node_ids)
            is_leaf = node_leaves[is_leaf] == current_node_ids            
            leaf_stops.append(is_leaf)
            p = points + t[:, np.newaxis] * directions
            child_ids = self._batch_find_child(centers, current_node_ids, p)
            parent_ids = (current_node_ids - 1) >> 3
            contains = self._batch_contains(centers, scales, p)
            
            # if is_leaf:
            #   t = tc_max + 1e-5
            # else:
            #   t = t
            _, tc_max = self._batch_intersect(bounds, points, dirs_inv)
            t = np.where(is_leaf, tc_max + 1e-5, t)

            # if is_leaf:
            #   id = parent
            # else:
            #   if contains(point):
            #       id = child
            #   else:
            #       id = parent
            #
            current_node_ids = np.where(contains & (~is_leaf), child_ids, parent_ids)

        return t_stops, node_stops

    def intersect(self, point: np.ndarray, direction: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_inv = np.reciprocal(direction)

        stack = [self.nodes[0]]
        stops = []
        t, t_max = stack[0].intersect_ray(point, dir_inv)
        t += 1e-5
        p = point + t * direction
        while stack:
            current = stack[-1]

            if current.id in self.leaves:
                _, tc_max = current.intersect_ray(point, dir_inv)
                stops.append((t, current.id))
                t = tc_max + 1e-5
                p = point + direction * t
                while current.contains(p):
                    t += 1e-5
                    p = point + direction * t
                    
                stack.pop()
                if t == t_max:
                    break
            else:
                if current.contains(p):
                    child_id = current.find_child(point + direction * t)
                    stack.append(self.nodes[child_id])
                else:
                    stack.pop()

        return stops
