"""Get the CREMI file, cut a small crop out of it, and clean the labels."""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

RAW_PATH = "volumes/raw"
LABELS_PATH = "volumes/labels/neuron_ids"
# CREMI voxel size in nanometres, order (z, y, x): slices are 40 nm apart, pixels are 4 nm.
DEFAULT_RESOLUTION_ZYX = (40.0, 4.0, 4.0)


def ensure_dataset(dest: Path, url: str, retries: int = 3, log: Callable[[str], None] = print) -> Path:
    """Make sure `dest` exists. Download it (to a temporary name first) if it does not."""
    dest = Path(dest)
    if dest.exists():
        log(f"Dataset found: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log(f"Downloading dataset (attempt {attempt}/{retries}) from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "em-viewer-prototype"})
            with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
                expected = response.headers.get("Content-Length")
                done = 0
                next_report = 25_000_000
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        log(f"  ...{done / 1e6:.0f} MB")
                        next_report += 25_000_000
            if expected is not None and tmp.stat().st_size != int(expected):
                raise IOError(f"download incomplete: got {tmp.stat().st_size} of {expected} bytes")
            if tmp.stat().st_size == 0:
                raise IOError("downloaded file is empty")
            os.replace(tmp, dest)
            log(f"Download finished: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
            return dest
        except Exception as error:  # noqa: BLE001 - we retry on anything, then report
            last_error = error
            log(f"  download failed: {error}")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"Could not download {url} after {retries} attempts ({last_error}). "
        f"You can also download it by hand and put it at {dest}."
    )


@dataclass
class Crop:
    raw: np.ndarray  # uint8, (z, y, x)
    labels: np.ndarray  # raw neuron ids as stored in the file, (z, y, x)
    resolution_zyx: tuple[float, float, float]  # nm
    start: tuple[int, int, int]  # actual crop start (z, y, x)
    full_shape: tuple[int, int, int]


def load_crop(hdf_path: Path, start: tuple[int, int, int], size: tuple[int, int, int]) -> Crop:
    """Read only the crop (not the whole 180 MB) from the HDF5 file."""
    import h5py  # imported here so the rest of the code can be tested without HDF5

    with h5py.File(hdf_path, "r") as f:
        for path in (RAW_PATH, LABELS_PATH):
            if path not in f:
                raise KeyError(f"{hdf_path} has no dataset {path!r}. Is it the right CREMI file?")
        raw_ds, label_ds = f[RAW_PATH], f[LABELS_PATH]
        if raw_ds.shape != label_ds.shape:
            raise ValueError(
                f"raw {raw_ds.shape} and labels {label_ds.shape} have different shapes; "
                "this prototype expects them to line up (use the non-padded sample file)."
            )
        shape = tuple(int(s) for s in raw_ds.shape)
        slices = []
        for axis, (s, n, full) in enumerate(zip(start, size, shape)):
            if s >= full:
                raise ValueError(f"CROP_START axis {axis} = {s} is outside the volume (size {full}).")
            slices.append(slice(s, min(s + n, full)))
        sl = tuple(slices)
        raw = np.asarray(raw_ds[sl])
        labels = np.asarray(label_ds[sl])
        res_attr = raw_ds.attrs.get("resolution")
        if res_attr is not None and len(res_attr) == 3:
            resolution = tuple(float(v) for v in res_attr)
        else:
            resolution = DEFAULT_RESOLUTION_ZYX
    return Crop(raw=raw, labels=labels, resolution_zyx=resolution, start=tuple(start), full_shape=shape)  # type: ignore[arg-type]


def clean_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Turn raw CREMI neuron ids into small numbers 1..K, with 0 meaning "no neuron".

    CREMI marks "no neuron" with the largest possible number of the label type
    (e.g. 2**64 - 1). Returns (uint32 array, K).
    """
    if np.issubdtype(labels.dtype, np.unsignedinteger):
        background = labels == np.iinfo(labels.dtype).max
    elif np.issubdtype(labels.dtype, np.signedinteger):
        background = labels < 0
    else:
        raise TypeError(f"Labels must be integers, got {labels.dtype}")

    # Work one slice at a time: sorting the whole crop at once needs several
    # times its size in memory, which a small cloud machine does not have.
    flat_labels = labels.reshape(labels.shape[0], -1)
    flat_background = background.reshape(labels.shape[0], -1)
    out = np.zeros(labels.shape, dtype=np.uint32)
    flat_out = out.reshape(labels.shape[0], -1)  # a view: writing here fills `out`
    pieces = [np.unique(flat_labels[i][~flat_background[i]]) for i in range(flat_labels.shape[0])]
    pieces = [p for p in pieces if p.size]
    if not pieces:
        return out, 0
    unique = np.unique(np.concatenate(pieces))
    for i in range(flat_labels.shape[0]):
        keep = ~flat_background[i]
        flat_out[i][keep] = (np.searchsorted(unique, flat_labels[i][keep]) + 1).astype(np.uint32)
    return out, int(unique.size)
