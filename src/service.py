"""EditService: merge, split and undo. The brain of the app.

It knows nothing about Neuroglancer. A viewer (or later, a button panel)
just calls these methods and listens for changes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from graph import SupervoxelGraph
from store import EditStore


@dataclass
class EditResult:
    ok: bool
    message: str
    select: tuple[int, ...] = ()  # objects the viewers should select now
    warning: str | None = None
    edit_id: int | None = None
    changed: bool = False
    code: str | None = None  # machine-readable reason for a refusal, so callers can add their own hints


class EditService:
    def __init__(self, graph: SupervoxelGraph, store: EditStore):
        self.graph = graph
        self.store = store
        self._lock = threading.RLock()
        self._listeners: list[Callable[[EditResult], None]] = []
        self.version = 0
        self.replayed_events = self._replay()

    # ----- start-up -------------------------------------------------------
    def _replay(self) -> int:
        """Apply every saved glue change on top of the starting glue."""
        self.graph.reset()
        events = self.store.live_events()
        for _, op, a, b in events:
            self._apply(op, a, b)
        self.graph.recompute()
        return len(events)

    def _apply(self, op: str, a: int, b: int) -> None:
        if op == "add":
            self.graph.add_edge(a, b)
        elif op == "remove":
            self.graph.remove_edge(a, b)
        else:
            raise ValueError(f"Unknown op {op!r}")

    # ----- helpers --------------------------------------------------------
    def subscribe(self, listener: Callable[[EditResult], None]) -> None:
        self._listeners.append(listener)

    def _commit(
        self,
        user: str,
        events: list[tuple[str, int, int]],
        result: EditResult,
        select: Callable[[], tuple[int, ...]] = lambda: (),
    ) -> EditResult:
        # 1) save first, so a crash never loses an edit that was shown
        result.edit_id = self.store.append_edit(user, events)
        # 2) change the live graph
        for op, a, b in events:
            self._apply(op, a, b)
        self.graph.recompute()
        self.version += 1
        result.changed = True
        result.select = tuple(select())  # worked out with the new numbers, before anyone is told
        # 3) tell the viewers to redraw
        for listener in list(self._listeners):
            listener(result)
        return result

    def brick_at(self, z: int, y: int, x: int) -> int:
        with self._lock:
            return self.graph.brick_at(z, y, x)

    def object_of(self, brick: int) -> int:
        with self._lock:
            return self.graph.object_of(brick)

    def existing_objects(self) -> set[int]:
        with self._lock:
            return set(int(o) for o in self.graph.object_ids())

    def lut_copy(self):
        with self._lock:
            return self.graph.lut.copy()

    # ----- edits ----------------------------------------------------------
    def merge(self, obj_a: int, obj_b: int, user: str = "unknown") -> EditResult:
        with self._lock:
            if obj_a == obj_b:
                return EditResult(False, "Pick two different objects to merge.")
            existing = self.existing_objects()
            if obj_a not in existing or obj_b not in existing:
                return EditResult(False, "One of the selected objects does not exist (any more).")
            a, b, touching = self.graph.best_merge_pair(obj_a, obj_b)
            warning = None if touching else "These objects do not touch; merged anyway using their closest bricks."
            result = EditResult(True, f"Merged {obj_a} and {obj_b}.", warning=warning)
            return self._commit(user, [("add", a, b)], result, select=lambda: (self.graph.object_of(a),))

    def split(self, bricks_a: list[int], bricks_b: list[int], user: str = "unknown") -> EditResult:
        with self._lock:
            if not bricks_a or not bricks_b:
                return EditResult(False, "Add at least one A seed and one B seed first.")
            objects = {self.graph.object_of(b) for b in list(bricks_a) + list(bricks_b)}
            if 0 in objects:
                return EditResult(False, "A seed is not on a cell. Move the mouse onto the cell and try again.")
            if len(objects) != 1:
                return EditResult(False, "All seeds must be on the same object.")
            if set(bricks_a) & set(bricks_b):
                return EditResult(
                    False,
                    "An A seed and a B seed fall in the same small piece. Move one of them further apart.",
                    code="seeds_overlap",
                )
            obj = next(iter(objects))
            cut = self.graph.find_cut(bricks_a, bricks_b)
            if not cut:  # cannot happen for a connected object, but never save an empty edit
                return EditResult(False, "Nothing to cut between these seeds.")
            result = EditResult(True, f"Split object {obj} (cut {len(cut)} connection{'s' if len(cut) != 1 else ''}).")
            return self._commit(
                user,
                [("remove", a, b) for a, b in cut],
                result,
                select=lambda: tuple(sorted({self.graph.object_of(bricks_a[0]), self.graph.object_of(bricks_b[0])})),
            )

    def undo(self, user: str = "unknown") -> EditResult:
        with self._lock:
            undone = self.store.undo_last()
            if undone is None:
                return EditResult(False, "Nothing to undo.")
            edit_id, author = undone
            self._replay()  # rebuild from the starting glue + remaining edits
            self.version += 1
            result = EditResult(True, f"Undid edit {edit_id} (made by {author}).", edit_id=edit_id, changed=True)
            for listener in list(self._listeners):
                listener(result)
            return result
