from collections import namedtuple
from itertools import product
from typing import List, Sequence

import numpy as np

CORNERS = np.array(list(product([-1, 1], [-1, 1], [-1, 1])), np.float32)

class OcTree:
    class Node(namedtuple("Node", ["id", "scale", "center", "parent"])):
        @property
        def child_ids(self) -> Sequence[int]:           
            start = self.id * 8 + 1
            return range(start, start + 8)

        def make_children(self) -> Sequence["OcTree.Node"]:
            scale = self.scale + 1
            offset = pow(2, -scale) * CORNERS
            for i, child in enumerate(self.child_ids):
                yield OcTree.Node(child, scale, self.center + offset[i], self.id)

    def __init__(self, initial_depth: int, max_nodes: int):
        self.max_nodes = max_nodes
        self.nodes = {
            0: OcTree.Node(0, 0, np.zeros(3, np.float32), None)
        }

        self.leaves: List[OcTree.Node] = []
        self.branches: List[OcTree.Node] = []

        stack = [self.nodes[0]]
        
        while stack:
            current = stack.pop()
            for child in current.make_children():
                self.nodes[child.id] = child
                if child.scale == initial_depth - 1:
                    self.leaves.append(child.id)
                else:
                    stack.append(child)
                    if child.scale == initial_depth - 2:
                        self.branches.append(child.id)

    def leaf_centers(self):
        return np.stack([self.nodes[i].center for i in self.leaves])

    def leaf_scales(self):
        return np.stack([self.nodes[i].scale for i in self.leaves])

    def split(self, leaf_id: int):
        current = self.nodes[leaf_id]
        for child in current.make_children():
            assert child.id not in self.nodes
            self.nodes[child.id] = child
            self.leaves.append(child.id)

        self.leaves.remove(leaf_id)
        self.branches.append(leaf_id)

    def merge(self, branch_id: int):
        current = self.nodes[branch_id]
        for child_id in current.child_ids:
            assert child_id in self.nodes
            del self.nodes[child_id]
            self.leaves.remove(child_id)

        self.leaves.append(branch_id)
        self.branches.remove(branch_id)

    def merge_and_split(self, leaf_values: np.ndarray, threshold: float) -> np.ndarray:
        leaf_opacity = {}
        branch_opacity = {}

        for leaf_id, value in zip(self.leaves, leaf_values):
            leaf_opacity[leaf_id] = value
            leaf = self.nodes[leaf_id]
            if not leaf.parent in branch_opacity:
                branch_opacity[leaf.parent] = 0
            
            branch_opacity[leaf.parent] += value / 8

        # sort branches by uncertainty (least to most certain)
        branch_values = np.array([branch_opacity[i] for i in self.branches])
        branch_values = np.abs(branch_values - 0.5)
        sorted_branch_ids = np.argsort(branch_values)
        sorted_branch_ids = [self.branches[idx] for idx in reversed(sorted_branch_ids)]

        # sort leaves by certainty (least to most uncertain)
        leaf_values = np.abs(leaf_values - 0.5)
        sorted_leaf_ids = np.argsort(leaf_values)
        sorted_leaf_ids = [self.leaves[idx] for idx in sorted_leaf_ids]

        # determine how many merge/splits we can make while
        # maintaining our node budget
        num_merge = (branch_values > 0.5 - threshold).sum()
        num_split = (leaf_values < threshold).sum()
        budget = self.max_nodes - len(self.nodes)
        num_split = min(num_split, num_merge + (budget // 8))

        print("Performing", num_merge, "merges and", num_split, "splits")
        for branch_id in sorted_branch_ids[:num_merge]:
            self.merge(branch_id)

        for leaf_id in sorted_leaf_ids[:num_split]:
            self.split(leaf_id)

        print("Num nodes:", len(self.nodes))

        # compute the updated opacity values
        opacity = []
        for leaf_id in self.leaves:
            node = self.nodes[leaf_id]
            if node.parent in leaf_opacity:
                opacity.append(leaf_opacity[node.parent])
            elif leaf_id in leaf_opacity:
                opacity.append(leaf_opacity[leaf_id])
            elif leaf_id in branch_opacity:
                opacity.append(branch_opacity[leaf_id])

        leaf_set = set(self.leaves)
        self.branches.clear()
        for node in self.nodes.values():
            if node.id in leaf_set:
                continue

            is_branch = True
            for child_id in node.child_ids:
                if child_id not in leaf_set:
                    is_branch = False
                    break
            
            if is_branch:
                self.branches.append(node.id)

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
        