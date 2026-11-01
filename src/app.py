"""Entry point: python app.py

Start-up order (PLAN.md section 6): settings -> data file -> crop -> bricks and glue
-> saved edits -> viewers -> print links.
"""

from __future__ import annotations

import gc
import logging
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Settings  # noqa: E402
from data import clean_labels, ensure_dataset, load_crop  # noqa: E402
from graph import SupervoxelGraph, build_model, make_fingerprint  # noqa: E402
from service import EditService  # noqa: E402
from store import EditStore  # noqa: E402

log = logging.getLogger("app")


def build_service(settings: Settings, crop_labels, resolution_zyx, crop_start) -> tuple[EditService, str]:
    cleaned, neurons = clean_labels(crop_labels)
    log.info("Crop has %d neurons; %.1f%% of voxels are 'no neuron'", neurons, 100.0 * (cleaned == 0).mean())
    model = build_model(cleaned, settings.block_size, resolution_zyx)
    del cleaned
    gc.collect()
    log.info("Built %d bricks and %d touching pairs", model.n_bricks, model.contact_a.size)

    fingerprint = make_fingerprint(model, crop_start, tuple(model.bricks.shape), settings.block_size)
    store, status = EditStore.open(settings.db_path, fingerprint)
    if status == "reset":
        log.warning(
            "Saved edits were made for different data (crop or block settings changed). "
            "The old edit log was moved aside next to %s and we start with no edits.",
            settings.db_path,
        )
    service = EditService(SupervoxelGraph(model), store)
    log.info("Replayed %d saved glue changes (%s); %d objects", service.replayed_events, status, service.graph.object_count())
    return service, status


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for warning in settings.short_token_warnings():
        log.warning(warning)

    ensure_dataset(settings.hdf_path, settings.dataset_url, log=log.info)
    crop = load_crop(settings.hdf_path, settings.crop_start, settings.crop_size)
    log.info(
        "Loaded crop %s from volume %s starting at %s; voxel size (z,y,x) = %s nm",
        crop.raw.shape, crop.full_shape, crop.start, crop.resolution_zyx,
    )
    service, _ = build_service(settings, crop.labels, crop.resolution_zyx, crop.start)
    raw = crop.raw
    resolution = crop.resolution_zyx
    del crop
    gc.collect()

    from viewer import ViewerApp  # imported late so tests do not need Neuroglancer

    app = ViewerApp(
        service=service,
        raw_zyx=raw,
        voxel_size_zyx=resolution,
        public_url=settings.public_url,
        port=settings.port,
        tokens=settings.tokens,
        mode=settings.viewer_mode,
        block_size=settings.block_size,
        mesh_simplify=settings.mesh_simplify,
    )
    links = app.start()
    print("=" * 70, flush=True)
    print(f"EM viewer ready ({settings.viewer_mode} mode). Open one link per person:", flush=True)
    for label, url in links:
        print(f"  {label}: {url}", flush=True)
    if settings.tokens_were_generated:
        print("  (tokens were generated at random; set VIEWER_TOKENS to keep the same links after a restart)", flush=True)
    print("=" * 70, flush=True)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()  # the Neuroglancer server runs in background threads; we just keep the process alive
    log.info("Shutting down")
    service.store.close()


if __name__ == "__main__":
    main()
