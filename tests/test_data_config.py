import numpy as np
import pytest

from config import Settings
from data import clean_labels, ensure_dataset


def test_clean_labels_turns_background_into_zero_and_renumbers():
    bg = np.uint64(2**64 - 1)
    raw = np.array([[bg, 500, 500], [7, bg, 90000]], dtype=np.uint64)
    out, k = clean_labels(raw)
    assert out.dtype == np.uint32 and k == 3
    assert out.tolist() == [[0, 2, 2], [1, 0, 3]]


def test_clean_labels_all_background_and_signed():
    out, k = clean_labels(np.full((2, 2, 2), np.uint64(2**64 - 1)))
    assert k == 0 and not out.any()
    out, k = clean_labels(np.array([-1, 5, 5, 9], dtype=np.int64))
    assert out.tolist() == [0, 1, 1, 2] and k == 2


def test_clean_labels_rejects_floats():
    with pytest.raises(TypeError):
        clean_labels(np.zeros(3, dtype=np.float32))


def test_download_then_skip(tmp_path):
    source = tmp_path / "src.bin"
    source.write_bytes(b"hello" * 1000)
    dest = tmp_path / "data" / "file.hdf"
    messages = []
    ensure_dataset(dest, source.as_uri(), log=messages.append)
    assert dest.read_bytes() == source.read_bytes()
    assert not list(dest.parent.glob("*.part"))
    source.unlink()  # a second start must not need the network
    assert ensure_dataset(dest, source.as_uri(), log=messages.append) == dest


def test_failed_download_leaves_nothing_half_written(tmp_path):
    dest = tmp_path / "file.hdf"
    with pytest.raises(RuntimeError, match="download it by hand|by hand"):
        ensure_dataset(dest, (tmp_path / "missing.bin").as_uri(), retries=1, log=lambda m: None)
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_settings_defaults(monkeypatch):
    for name in ("DATA_DIR", "PORT", "PUBLIC_URL", "VIEWER_MODE", "VIEWER_TOKENS", "CROP_START", "CROP_SIZE", "BLOCK_SIZE"):
        monkeypatch.delenv(name, raising=False)
    s = Settings.from_env()
    assert s.port == 8080 and s.public_url == "http://localhost:8080"
    assert s.viewer_mode == "per_user" and len(s.tokens) == 3 and s.tokens_were_generated
    assert s.crop_size == (64, 512, 512) and s.block_size == (8, 64, 64) and s.crop_start == (30, 300, 300)
    assert str(s.db_path).endswith("edits.sqlite")
    assert s.mesh_simplify is False


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("PUBLIC_URL", "https://x.fly.dev/")
    monkeypatch.setenv("VIEWER_MODE", "shared")
    monkeypatch.setenv("VIEWER_TOKENS", "alice-1234567890, bob")
    monkeypatch.setenv("CROP_SIZE", "16,128,128")
    s = Settings.from_env()
    assert s.public_url == "https://x.fly.dev" and s.tokens == ("alice-1234567890", "bob")
    assert s.crop_size == (16, 128, 128) and not s.tokens_were_generated
    assert len(s.short_token_warnings()) == 1


def test_mesh_simplify_setting(monkeypatch):
    monkeypatch.setenv("MESH_SIMPLIFY", "On")
    assert Settings.from_env().mesh_simplify is True
    monkeypatch.setenv("MESH_SIMPLIFY", "false")
    assert Settings.from_env().mesh_simplify is False


@pytest.mark.parametrize(
    "name,value",
    [("VIEWER_MODE", "everyone"), ("CROP_SIZE", "1,2"), ("CROP_SIZE", "0,5,5"), ("VIEWER_TOKENS", "a b"), ("VIEWER_TOKENS", "aa,aa"), ("MESH_SIMPLIFY", "maybe")],
)
def test_settings_reject_bad_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()
