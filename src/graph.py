"""Bricks (supervoxels), glue (edges) and objects (connected groups of bricks).

No Neuroglancer in this file, so it is easy to test.
All arrays here use (z, y, x) order, like the HDF5 file. Lengths are in nanometres.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

Pair = tuple[int, int]


def make_pair(a: int, b: int) -> Pair:
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------- #
# Step 1: cut the neurons into bricks
# --------------------------------------------------------------------------- #
def build_bricks(labels: np.ndarray, block_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cut every neuron into bricks.

    A brick is a connected piece of one neuron that sits inside one block of
    `block_size`. Returns (brick picture uint32, neuron_of) where
    neuron_of[brick] is the neuron label (neuron_of[0] = 0 = no neuron).
    """
    if labels.ndim != 3:
        raise ValueError("labels must be a 3D array")
    bricks = np.zeros(labels.shape, dtype=np.uint32)
    neuron_of: list[int] = [0]
    next_id = 1
    bz, by, bx = block_size
    for z0 in range(0, labels.shape[0], bz):
        for y0 in range(0, labels.shape[1], by):
            for x0 in range(0, labels.shape[2], bx):
                sl = (slice(z0, z0 + bz), slice(y0, y0 + by), slice(x0, x0 + bx))
                sub = labels[sl]
                out = bricks[sl]  # a view: writing into it fills `bricks`
                for neuron in np.unique(sub):
                    if neuron == 0:
                        continue
                    pieces, count = ndimage.label(sub == neuron)  # touching by faces (6-connected)
                    inside = pieces > 0
                    out[inside] = pieces[inside].astype(np.uint32) + np.uint32(next_id - 1)
                    neuron_of.extend([int(neuron)] * count)
                    next_id += count
    return bricks, np.asarray(neuron_of, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Step 2: find where bricks touch
# --------------------------------------------------------------------------- #
@dataclass
class BrickModel:
    """Everything we know about the bricks that never changes after start-up."""

    bricks: np.ndarray  # (z, y, x) uint32, 0 = no neuron
    neuron_of: np.ndarray  # (N+1,) neuron label of each brick, index 0 = 0
    voxel_count: np.ndarray  # (N+1,)
    centroid_nm: np.ndarray  # (N+1, 3) centre of each brick, nm, (z, y, x)
    voxel_size_nm: tuple[float, float, float]  # (z, y, x)
    # Every place where two different bricks touch (a < b):
    contact_a: np.ndarray
    contact_b: np.ndarray
    contact_area: np.ndarray  # nm^2
    contact_pos_nm: np.ndarray  # (K, 3) average touching point, nm
    contact_same: np.ndarray  # True when both bricks belong to the same neuron (= starting glue)

    @property
    def n_bricks(self) -> int:
        return int(self.neuron_of.size - 1)


def build_model(
    labels: np.ndarray,
    block_size: tuple[int, int, int],
    voxel_size_nm: tuple[float, float, float],
) -> BrickModel:
    bricks, neuron_of = build_bricks(labels, block_size)
    n = int(neuron_of.size)  # N + 1

    # Size and centre of each brick, one slice at a time to keep memory small.
    counts = np.zeros(n, dtype=np.float64)
    sums = np.zeros((n, 3), dtype=np.float64)
    yy, xx = np.indices(bricks.shape[1:])
    yy, xx = yy.ravel().astype(np.float64), xx.ravel().astype(np.float64)
    for z in range(bricks.shape[0]):
        flat = bricks[z].ravel()
        c = np.bincount(flat, minlength=n).astype(np.float64)
        counts += c
        sums[:, 0] += c * (z + 0.5)
        sums[:, 1] += np.bincount(flat, weights=yy + 0.5, minlength=n)
        sums[:, 2] += np.bincount(flat, weights=xx + 0.5, minlength=n)
    safe = np.maximum(counts, 1.0)[:, None]
    centroid_nm = (sums / safe) * np.asarray(voxel_size_nm)[None, :]

    ca, cb, area, pos = _find_contacts(bricks, n, voxel_size_nm)
    return BrickModel(
        bricks=bricks,
        neuron_of=neuron_of,
        voxel_count=counts.astype(np.int64),
        centroid_nm=centroid_nm,
        voxel_size_nm=tuple(float(v) for v in voxel_size_nm),  # type: ignore[arg-type]
        contact_a=ca,
        contact_b=cb,
        contact_area=area,
        contact_pos_nm=pos,
        contact_same=neuron_of[ca] == neuron_of[cb],
    )


def _find_contacts(bricks: np.ndarray, n: int, voxel_size_nm) -> tuple[np.ndarray, ...]:
    keys, areas, poses = [], [], []
    vs = np.asarray(voxel_size_nm, dtype=np.float64)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a, b = bricks[tuple(lo)], bricks[tuple(hi)]
        mask = (a != b) & (a > 0) & (b > 0)
        idx = np.nonzero(mask)
        if idx[0].size == 0:
            continue
        aa, bb = a[mask].astype(np.int64), b[mask].astype(np.int64)
        keys.append(np.minimum(aa, bb) * n + np.maximum(aa, bb))
        # A touching face is as big as the other two voxel sides multiplied.
        face = float(np.prod(np.delete(vs, axis)))
        areas.append(np.full(aa.size, face))
        pos = np.stack(idx, axis=1).astype(np.float64) + 0.5
        pos[:, axis] += 0.5  # the touching point lies between the two voxel centres
        poses.append(pos * vs[None, :])
    if not keys:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros((0, 3))
    keys, areas, poses = np.concatenate(keys), np.concatenate(areas), np.concatenate(poses)
    unique, inverse = np.unique(keys, return_inverse=True)
    area = np.bincount(inverse, weights=areas, minlength=unique.size)
    pos = np.stack(
        [np.bincount(inverse, weights=areas * poses[:, k], minlength=unique.size) / area for k in range(3)],
        axis=1,
    )
    return unique // n, unique % n, area, pos


# --------------------------------------------------------------------------- #
# Step 3: the live graph (bricks + glue), and the cheat sheet
# --------------------------------------------------------------------------- #
class SupervoxelGraph:
    """Bricks joined by glue. Connected groups of bricks are the objects.

    `lut` is the cheat sheet: lut[brick] = object number (0 = no neuron).
    The object number of a group is the smallest brick number inside it.
    """

    # Glue added by hand on bricks that do not really touch is the weakest glue,
    # so a later split prefers to undo it.
    MANUAL_EDGE_WEIGHT = 1.0

    def __init__(self, model: BrickModel):
        self.model = model
        self.n = int(model.neuron_of.size)  # N + 1 (index 0 is "no neuron")
        self._contact: dict[Pair, float] = {
            (int(a), int(b)): float(w) for a, b, w in zip(model.contact_a, model.contact_b, model.contact_area)
        }
        self.baseline: dict[Pair, float] = {
            (int(a), int(b)): float(w)
            for a, b, w, same in zip(model.contact_a, model.contact_b, model.contact_area, model.contact_same)
            if same
        }
        self.edges: dict[Pair, float] = {}
        self.adj: dict[int, set[int]] = {}
        self.lut = np.zeros(self.n, dtype=np.uint32)
        self.reset()

    # ----- state ----------------------------------------------------------
    def reset(self) -> None:
        """Go back to the starting glue (before any edit)."""
        self.edges = dict(self.baseline)
        self._rebuild_adjacency()
        self.recompute()

    def _rebuild_adjacency(self) -> None:
        self.adj = {}
        for a, b in self.edges:
            self.adj.setdefault(a, set()).add(b)
            self.adj.setdefault(b, set()).add(a)

    def add_edge(self, a: int, b: int) -> float:
        pair = make_pair(a, b)
        if not (0 < pair[0] < self.n and 0 < pair[1] < self.n) or pair[0] == pair[1]:
            raise ValueError(f"Bad brick pair {pair}")
        weight = self._contact.get(pair, self.MANUAL_EDGE_WEIGHT)
        self.edges[pair] = weight
        self.adj.setdefault(pair[0], set()).add(pair[1])
        self.adj.setdefault(pair[1], set()).add(pair[0])
        return weight

    def remove_edge(self, a: int, b: int) -> bool:
        pair = make_pair(a, b)
        if pair not in self.edges:
            return False
        del self.edges[pair]
        self.adj[pair[0]].discard(pair[1])
        self.adj[pair[1]].discard(pair[0])
        return True

    def recompute(self) -> None:
        """Rebuild the cheat sheet from the current glue."""
        if self.edges:
            pairs = np.fromiter((v for pair in self.edges for v in pair), dtype=np.int64).reshape(-1, 2)
            rows, cols = pairs[:, 0], pairs[:, 1]
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
        matrix = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(self.n, self.n))
        count, component = connected_components(matrix, directed=False)
        smallest = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(smallest, component[1:], np.arange(1, self.n))
        lut = smallest[component].astype(np.uint32)
        lut[0] = 0
        self.lut = lut

    # ----- questions ------------------------------------------------------
    def object_ids(self) -> np.ndarray:
        return np.unique(self.lut[1:])

    def object_count(self) -> int:
        return int(self.object_ids().size)

    def members(self, obj: int) -> np.ndarray:
        return np.flatnonzero(self.lut == obj)

    def object_of(self, brick: int) -> int:
        return int(self.lut[brick])

    def brick_at(self, z: int, y: int, x: int) -> int:
        shape = self.model.bricks.shape
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            return 0
        return int(self.model.bricks[z, y, x])

    # ----- merge ----------------------------------------------------------
    def best_merge_pair(self, obj_a: int, obj_b: int) -> tuple[int, int, bool]:
        """Pick two bricks to glue together. Returns (brick_in_a, brick_in_b, really_touching)."""
        m = self.model
        oa, ob = self.lut[m.contact_a], self.lut[m.contact_b]
        forward = (oa == obj_a) & (ob == obj_b)
        backward = (oa == obj_b) & (ob == obj_a)
        touching = forward | backward
        if touching.any():
            candidates = np.flatnonzero(touching)
            best = candidates[np.argmax(m.contact_area[candidates])]
            a, b = int(m.contact_a[best]), int(m.contact_b[best])
            return (a, b, True) if self.lut[a] == obj_a else (b, a, True)

        mem_a, mem_b = self.members(obj_a), self.members(obj_b)
        if mem_a.size == 0 or mem_b.size == 0:
            raise ValueError("One of the objects does not exist")
        tree = cKDTree(m.centroid_nm[mem_b])
        distance, nearest = tree.query(m.centroid_nm[mem_a])
        i = int(np.argmin(distance))
        return int(mem_a[i]), int(mem_b[nearest[i]]), False

    # ----- split ----------------------------------------------------------
    def find_cut(self, bricks_a: list[int], bricks_b: list[int]) -> list[Pair]:
        """Thinnest set of glue that separates the A bricks from the B bricks.

        All bricks must be in one object, and no brick may be on both sides.
        Returns the list of glue pairs to remove.
        """
        objects = {self.object_of(b) for b in list(bricks_a) + list(bricks_b)}
        if len(objects) != 1 or 0 in objects:
            raise ValueError("All seeds must sit on the same object")
        if set(bricks_a) & set(bricks_b):
            raise ValueError("A seed and a B seed are in the same brick")
        obj = objects.pop()
        members = set(int(b) for b in self.members(obj))

        graph = nx.Graph()
        for a in members:
            for b in self.adj.get(a, ()):
                if a < b:
                    graph.add_edge(a, b, capacity=self.edges[(a, b)])
        start, end = "start", "end"  # hidden nodes, joined by unbreakable glue (no capacity = infinite)
        for a in bricks_a:
            graph.add_edge(start, int(a))
        for b in bricks_b:
            graph.add_edge(end, int(b))

        _, (side_a, side_b) = nx.minimum_cut(graph, start, end)
        cut = [
            make_pair(a, b)
            for a in side_a
            if a != start
            for b in self.adj.get(a, ())
            if b in side_b and b != end
        ]
        return sorted(set(cut))


def make_fingerprint(
    model: BrickModel,
    crop_start: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
) -> dict:
    """A short description of the starting bricks and glue.

    Saved edits only make sense for exactly these bricks, so we compare this
    on start-up before replaying the log.
    """
    return {
        "crop_start": list(crop_start),
        "crop_size": list(crop_size),
        "block_size": list(block_size),
        "voxel_size_nm": list(model.voxel_size_nm),
        "shape": list(model.bricks.shape),
        "n_bricks": model.n_bricks,
        "n_glue": int(model.contact_same.sum()),
        "brick_hash": hashlib.sha1(np.ascontiguousarray(model.bricks).tobytes()).hexdigest(),
    }
