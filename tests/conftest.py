import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import clean_labels  # noqa: E402
from graph import SupervoxelGraph, build_model, make_fingerprint  # noqa: E402
from service import EditService  # noqa: E402
from store import EditStore  # noqa: E402

BACKGROUND = np.uint64(2**64 - 1)
VOXEL_SIZE = (40.0, 4.0, 4.0)  # (z, y, x) nm, like CREMI
BLOCK = (6, 16, 10)
CROP = ((0, 0, 0), (6, 16, 40))


def make_labels():
    """A tiny made-up brain, (z, y, x) = (6, 16, 40).

    neuron A (raw id 100): a thick bar along x with a thin neck at x=18..21
    neuron B (raw id 205): a bar that does NOT touch A
    neuron C (raw id 7):   a short bar between them; touches both A and B
    """
    lab = np.full((6, 16, 40), BACKGROUND, dtype=np.uint64)
    lab[:, 2:9, :] = 100
    lab[:, 2:9, 18:22] = BACKGROUND
    lab[2:4, 5:7, 18:22] = 100  # the thin neck
    lab[:, 11:15, :] = 205
    lab[:, 9:11, 0:11] = 7
    return lab


@pytest.fixture
def model():
    cleaned, _ = clean_labels(make_labels())
    return build_model(cleaned, BLOCK, VOXEL_SIZE)


@pytest.fixture
def fingerprint(model):
    return make_fingerprint(model, CROP[0], CROP[1], BLOCK)


@pytest.fixture
def graph(model):
    return SupervoxelGraph(model)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "edits.sqlite"


@pytest.fixture
def svc(model, fingerprint, db_path):
    store, _ = EditStore.open(db_path, fingerprint)
    return EditService(SupervoxelGraph(model), store)


# Handy spots (z, y, x) on each neuron
A_LEFT, A_RIGHT = (3, 5, 2), (3, 5, 35)
A_MID_LEFT, A_MID_RIGHT = (3, 5, 15), (3, 5, 25)  # the two bricks either side of the neck
B_SPOT = (3, 12, 2)
C_SPOT = (3, 9, 2)
