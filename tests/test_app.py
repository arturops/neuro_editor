"""Start-up wiring (app.build_service) on the made-up brain: edit, 'restart', check."""
import numpy as np

import app
from config import Settings
from conftest import A_LEFT, A_RIGHT, B_SPOT, C_SPOT, make_labels


def settings_for(tmp_path, monkeypatch, block="6,16,10"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BLOCK_SIZE", block)
    return Settings.from_env()


def test_edits_survive_a_restart(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, monkeypatch)
    service, status = app.build_service(settings, make_labels(), (40.0, 4.0, 4.0), (0, 0, 0))
    assert status == "new"
    brick = service.brick_at
    service.merge(service.object_of(brick(*A_LEFT)), service.object_of(brick(*C_SPOT)), "user1")
    service.split([brick(*A_LEFT)], [brick(*A_RIGHT)], "user2")
    expected = service.lut_copy()
    service.store.close()

    again, status = app.build_service(settings, make_labels(), (40.0, 4.0, 4.0), (0, 0, 0))
    assert status == "match" and again.replayed_events >= 2
    assert (again.lut_copy() == expected).all()


def test_changed_block_size_starts_fresh_and_keeps_the_old_log(tmp_path, monkeypatch):
    service, _ = app.build_service(settings_for(tmp_path, monkeypatch), make_labels(), (40.0, 4.0, 4.0), (0, 0, 0))
    service.merge(service.object_of(service.brick_at(*A_LEFT)), service.object_of(service.brick_at(*B_SPOT)), "user1")
    service.store.close()

    other, status = app.build_service(settings_for(tmp_path, monkeypatch, block="6,16,20"), make_labels(), (40.0, 4.0, 4.0), (0, 0, 0))
    assert status == "reset" and other.replayed_events == 0
    assert list(tmp_path.glob("edits.sqlite.mismatch-*"))
