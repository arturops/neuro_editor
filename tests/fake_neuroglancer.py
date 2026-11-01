"""A tiny stand-in for the `neuroglancer` package, so we can test our own viewer code
(key handling, seeds, selection, redraw) without installing Neuroglancer.

It copies only the parts of the real API that viewer.py uses. It cannot prove that the
real package behaves the same way; that has to be checked with the real package.
"""
from types import SimpleNamespace
from contextlib import contextmanager

bind_calls = []


def set_server_bind_address(bind_address="127.0.0.1", bind_port=0):
    bind_calls.append((bind_address, bind_port))


class CoordinateSpace:
    def __init__(self, names, units, scales):
        self.names, self.units, self.scales = names, units, scales


class LocalVolume:
    created = 0  # how many volumes exist so far; a swap of the layer's source creates a new one

    def __init__(self, data, dimensions=None, volume_type=None):
        LocalVolume.created += 1
        self.data, self.dimensions, self.volume_type = data, dimensions, volume_type
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1


class ImageLayer:
    def __init__(self, source):
        self.source = source


class SegmentationLayer:
    def __init__(self, source):
        self.source = source
        self.segments = set()


class LocalAnnotationLayer:
    def __init__(self, dimensions=None, annotations=None, annotation_color=None):
        self.dimensions, self.annotations, self.annotation_color = dimensions, annotations or [], annotation_color


class EllipsoidAnnotation:
    def __init__(self, id, center, radii):
        self.id, self.center, self.radii = id, center, radii


class _Actions:
    def __init__(self):
        self.handlers = {}

    def add(self, name, handler):
        self.handlers[name] = handler


class _ConfigState:
    def __init__(self):
        self.status_messages = {}
        self.input_event_bindings = SimpleNamespace(viewer={}, slice_view={}, perspective_view={})

    @contextmanager
    def txn(self):
        yield self


class Viewer:
    def __init__(self, token="auto"):
        self.token = token
        self.state = SimpleNamespace(layers={}, dimensions=None, layout=None)
        self.actions = _Actions()
        self.config_state = _ConfigState()

    @contextmanager
    def txn(self):
        yield self.state

    def get_viewer_url(self):
        return f"http://0.0.0.0:8080/v/{self.token}/"
