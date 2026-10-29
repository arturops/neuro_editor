"""Neuroglancer layers, keyboard actions and per-user viewers.

DESIGN NOTE (prototype): The crop is held in memory and served via
neuroglancer.LocalVolume, with meshes generated on demand by Neuroglancer.
Fine for a <=3-user demo; it does not scale beyond a single process/crop.

To scale up:
 1. Write EM + segmentation as Neuroglancer "precomputed" chunks
    (cloud-volume / tensorstore) on object storage (e.g. Tigris on Fly, S3).
 2. Keep supervoxel labels immutable; apply edits only as a
    supervoxel->object mapping (the "agglomeration"), never rewriting voxels.
 3. Move the graph + edit log from SQLite to Postgres, with a worker that
    recomputes meshes for the affected objects only.
 4. Production reference: CAVE / PyChunkedGraph with Neuroglancer's
    "graphene" source, which is this same design at scale.

This file is the only place that talks to Neuroglancer. Everything it does to
the data goes through EditService (service.py).
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse

import numpy as np

from service import EditResult, EditService

log = logging.getLogger("viewer")

EM_LAYER = "em"
SEG_LAYER = "segmentation"

# Action name -> key. Press `h` inside Neuroglancer to see its own keys and check for clashes.
KEYS = {
    "em-merge": "shift+keym",
    "em-seed-a": "shift+keya",
    "em-seed-b": "shift+keyb",
    "em-split": "shift+keyx",
    "em-clear-seeds": "shift+keyc",
    "em-undo": "shift+keyu",
}
SEED_LAYERS = {"A": "seeds-A", "B": "seeds-B"}
SEED_COLORS = {"A": "#000000", "B": "#ffffff" }
SEED_RADIUS_FRACTION = 0.2  # of the smallest block side, so a marker never outgrows its block
HELP_TEXT = (
    "Shift+M merge the 2 selected objects | Shift+A / Shift+B add seed A / B under the mouse | "
    "Shift+X split | Shift+C clear seeds | Shift+U undo"
)


@dataclass
class Session:
    """One Neuroglancer viewer plus the seeds its user has placed."""

    label: str  # shown in the edit log (never the secret token)
    token: str
    viewer: object
    seeds: dict[str, list[int]] = field(default_factory=lambda: {"A": [], "B": []})
    seed_points: dict[str, list[tuple[int, int, int]]] = field(
        default_factory=lambda: {"A": [], "B": []}
    )  # (x, y, z) voxels where each seed was placed, for the markers


class ViewerApp:
    def __init__(
        self,
        service: EditService,
        raw_zyx: np.ndarray,
        voxel_size_zyx: tuple[float, float, float],
        public_url: str,
        port: int,
        tokens: tuple[str, ...],
        mode: str,
        block_size: tuple[int, int, int] = (8, 64, 64),
        ng=None,
    ):
        self.ng = ng or importlib.import_module("neuroglancer")
        self.service = service
        self.public_url = public_url.rstrip("/")
        self.port = port
        self.mode = mode
        self.tokens = tokens if mode == "per_user" else tokens[:1]
        self._tls = threading.local()  # remembers which session is making an edit

        # Neuroglancer wants (x, y, z) order, matching the dimension names below.
        sz, sy, sx = voxel_size_zyx
        self.dims = self.ng.CoordinateSpace(names=["x", "y", "z"], units="nm", scales=[sx, sy, sz])
        # Seed marker radius per axis (x, y, z), in voxels of that axis: a true sphere in nm.
        bz, by, bx = block_size
        radius_nm = SEED_RADIUS_FRACTION * min(bz * sz, by * sy, bx * sx)
        self._seed_radii = [radius_nm / sx, radius_nm / sy, radius_nm / sz]
        self._bricks_xyz = service.graph.model.bricks.transpose(2, 1, 0)  # a view, no copy
        self._display = np.ascontiguousarray(service.graph.lut[self._bricks_xyz])  # what Neuroglancer shows
        em_xyz = np.ascontiguousarray(raw_zyx.transpose(2, 1, 0))
        self._em_volume = self.ng.LocalVolume(data=em_xyz, dimensions=self.dims, volume_type="image")
        self._seg_volume = self.ng.LocalVolume(data=self._display, dimensions=self.dims, volume_type="segmentation")

        self.sessions: list[Session] = []
        service.subscribe(self._on_change)

    # ----- start ----------------------------------------------------------
    def start(self) -> list[tuple[str, str]]:
        """Start the Neuroglancer server and create the viewers. Returns [(label, url)]."""
        # 0.0.0.0 so the cloud proxy can reach us; the port comes from $PORT.
        self.ng.set_server_bind_address(bind_address="0.0.0.0", bind_port=self.port)
        links = []
        for index, token in enumerate(self.tokens):
            label = f"user{index + 1}" if self.mode == "per_user" else "shared"
            viewer = self._new_viewer(token)
            session = Session(label=label, token=token, viewer=viewer)
            self._setup_layers(viewer)
            self._setup_actions(session)
            self.sessions.append(session)
            links.append((label, self._public_link(viewer)))
        return links

    def _new_viewer(self, token: str):
        try:
            return self.ng.Viewer(token=token)
        except TypeError:  # very old Neuroglancer: no fixed tokens, links change on restart
            log.warning("This Neuroglancer version does not accept fixed tokens; links will change on restart.")
            return self.ng.Viewer()

    def _public_link(self, viewer) -> str:
        """Neuroglancer thinks it lives on localhost; swap in our public address."""
        real = viewer.get_viewer_url()
        parsed = urlparse(real)
        log.info("Neuroglancer's own address for this viewer: %s", real)
        if parsed.path.startswith("/v/"):
            return f"{self.public_url}{parsed.path}"
        log.warning("Unexpected viewer URL shape %r; using it as is.", real)
        return real

    def _setup_layers(self, viewer) -> None:
        with viewer.txn() as s:
            s.dimensions = self.dims
            s.layers[EM_LAYER] = self.ng.ImageLayer(source=self._em_volume)
            s.layers[SEG_LAYER] = self.ng.SegmentationLayer(source=self._seg_volume)
            s.layout = "4panel"

    def _setup_actions(self, session: Session) -> None:
        handlers = {
            "em-merge": self._merge,
            "em-seed-a": lambda sess, st: self._add_seed(sess, st, "A"),
            "em-seed-b": lambda sess, st: self._add_seed(sess, st, "B"),
            "em-split": self._split,
            "em-clear-seeds": self._clear_seeds,
            "em-undo": self._undo,
        }
        viewer = session.viewer
        for name, handler in handlers.items():
            viewer.actions.add(name, self._guard(session, handler))
        with viewer.config_state.txn() as s:
            for name, key in KEYS.items():
                s.input_event_bindings.viewer[key] = name
                s.input_event_bindings.slice_view[key] = name
                s.input_event_bindings.perspective_view[key] = name
            s.status_messages["em-help"] = HELP_TEXT

    # ----- keyboard actions -----------------------------------------------
    def _guard(self, session: Session, handler):
        def run(action_state):
            self._tls.session = session
            try:
                handler(session, action_state)
            except Exception as error:  # noqa: BLE001 - never let a key press kill the server
                log.exception("Action failed")
                self._say(session, f"Something went wrong: {error}")
            finally:
                self._tls.session = None

        return run

    @staticmethod
    def _selected(action_state) -> list[int]:
        layer = action_state.viewer_state.layers[SEG_LAYER]
        return sorted(int(v) for v in layer.segments)

    def _merge(self, session: Session, action_state) -> None:
        selected = self._selected(action_state)
        if len(selected) != 2:
            self._say(session, f"Select exactly two objects first (you have {len(selected)} selected).")
            return
        self._report(session, self.service.merge(selected[0], selected[1], user=session.label))

    def _add_seed(self, session: Session, action_state, side: str) -> None:
        coords = getattr(action_state, "mouse_voxel_coordinates", None)
        if coords is None:
            self._say(session, "Hover the mouse over the picture, then press the key again.")
            return
        x, y, z = (int(np.floor(c)) for c in coords[:3])
        brick = self.service.brick_at(z, y, x)
        if brick == 0:
            self._say(session, "No cell under the mouse. Move onto a cell and try again.")
            return
        if brick not in session.seeds[side]:
            session.seeds[side].append(brick)
            session.seed_points[side].append((x, y, z))
            self._draw_seeds(session)
        obj = self.service.object_of(brick)
        others = {self.service.object_of(b) for b in session.seeds["A"] + session.seeds["B"]}
        note = "" if len(others) == 1 else "  WARNING: seeds are on different objects."
        self._say(
            session,
            f"Seed {side} on object {obj}. Now {len(session.seeds['A'])} A and {len(session.seeds['B'])} B seeds.{note}",
        )

    def _draw_seeds(self, session: Session) -> None:
        """Redraw the A / B marker layers (spheres, so they show in nearby 2D slices too)."""
        with session.viewer.txn() as s:
            for side, layer_name in SEED_LAYERS.items():
                points = session.seed_points[side]
                if not points:
                    if layer_name in s.layers:
                        del s.layers[layer_name]
                    continue
                s.layers[layer_name] = self.ng.LocalAnnotationLayer(
                    dimensions=self.dims,
                    annotation_color=SEED_COLORS[side],
                    annotations=[
                        self.ng.EllipsoidAnnotation(
                            id=f"{side}{i}", center=[x + 0.5, y + 0.5, z + 0.5], radii=self._seed_radii
                        )
                        for i, (x, y, z) in enumerate(points)
                    ],
                )

    def _reset_seeds(self, session: Session) -> None:
        session.seeds = {"A": [], "B": []}
        session.seed_points = {"A": [], "B": []}
        try:
            self._draw_seeds(session)
        except Exception:  # noqa: BLE001
            log.exception("Could not remove seed markers for %s", session.label)

    def _split(self, session: Session, action_state) -> None:
        result = self.service.split(session.seeds["A"], session.seeds["B"], user=session.label)
        if result.ok:
            self._reset_seeds(session)
        self._report(session, result)

    def _clear_seeds(self, session: Session, action_state) -> None:
        self._reset_seeds(session)
        self._say(session, "Seeds cleared.")

    def _undo(self, session: Session, action_state) -> None:
        self._report(session, self.service.undo(user=session.label))

    def _report(self, session: Session, result: EditResult) -> None:
        text = result.message + (f"  NOTE: {result.warning}" if result.warning else "")
        self._say(session, text)

    # ----- redraw after every edit ----------------------------------------
    def _on_change(self, result: EditResult) -> None:
        """Called by EditService (already holding its lock) after every saved change."""
        lut = self.service.graph.lut
        self._display[...] = lut[self._bricks_xyz]  # repaint from the cheat sheet
        self._seg_volume.invalidate()  # tells Neuroglancer to refetch pictures and re-make 3D meshes
        editor = getattr(self._tls, "session", None)
        for session in self.sessions:
            try:
                with session.viewer.txn() as s:
                    layer = s.layers[SEG_LAYER]
                    if session is editor and result.select:
                        layer.segments = set(result.select)
                    else:
                        # Follow each selected object through the edit: an object number is the
                        # number of its smallest brick, so that brick tells us where it went.
                        kept = {int(lut[o]) for o in (int(v) for v in layer.segments) if 0 < o < lut.size}
                        layer.segments = kept - {0}
            except Exception:  # noqa: BLE001
                log.exception("Could not update selection for %s", session.label)
            if session is not editor:
                self._reset_seeds(session)  # old seeds may no longer be meaningful

    # ----- small helper ---------------------------------------------------
    def _say(self, session: Session, text: str, seconds: float = 10.0) -> None:
        log.info("[%s] %s", session.label, text)
        key = "em-msg"
        try:
            with session.viewer.config_state.txn() as s:
                s.status_messages[key] = text

            def clear():
                try:
                    with session.viewer.config_state.txn() as s:
                        if s.status_messages.get(key) == text:
                            del s.status_messages[key]
                except Exception:  # noqa: BLE001
                    pass

            timer = threading.Timer(seconds, clear)
            timer.daemon = True
            timer.start()
        except Exception:  # noqa: BLE001
            log.exception("Could not show status message")
