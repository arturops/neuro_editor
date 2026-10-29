import numpy as np

from conftest import A_LEFT, A_MID_LEFT, A_MID_RIGHT, A_RIGHT, B_SPOT, C_SPOT
from graph import SupervoxelGraph, make_pair
from service import EditService
from store import EditStore


def obj(svc, spot):
    return svc.object_of(svc.brick_at(*spot))


def brick(svc, spot):
    return svc.brick_at(*spot)


def count(svc):
    return svc.graph.object_count()


def test_merge_touching_objects(svc):
    a, c = obj(svc, A_LEFT), obj(svc, C_SPOT)
    result = svc.merge(a, c, "alice")
    assert result.ok and result.warning is None
    assert count(svc) == 2
    assert obj(svc, A_LEFT) == obj(svc, C_SPOT)
    assert result.select == (obj(svc, A_LEFT),)


def test_merge_far_apart_objects_warns_but_works(svc):
    result = svc.merge(obj(svc, A_LEFT), obj(svc, B_SPOT), "alice")
    assert result.ok and result.warning and "do not touch" in result.warning
    assert count(svc) == 2


def test_merge_rejects_same_or_missing_objects(svc):
    a = obj(svc, A_LEFT)
    assert not svc.merge(a, a).ok
    assert not svc.merge(a, 99999).ok
    assert count(svc) == 3


def test_split_cuts_only_the_thin_neck(svc):
    assert svc.split([brick(svc, A_LEFT)], [brick(svc, A_RIGHT)], "alice").ok
    assert count(svc) == 4
    assert obj(svc, A_LEFT) != obj(svc, A_RIGHT)
    removed = [(a, b) for _, op, a, b in svc.store.live_events() if op == "remove"]
    assert removed == [make_pair(brick(svc, A_MID_LEFT), brick(svc, A_MID_RIGHT))]


def test_split_with_many_seeds(svc):
    a_seeds = [brick(svc, A_LEFT), brick(svc, (3, 5, 15))]
    b_seeds = [brick(svc, A_RIGHT), brick(svc, (1, 7, 27))]
    result = svc.split(a_seeds, b_seeds, "alice")
    assert result.ok
    assert obj(svc, A_LEFT) == obj(svc, (3, 5, 15)) != obj(svc, A_RIGHT)
    assert set(result.select) == {obj(svc, A_LEFT), obj(svc, A_RIGHT)}


def test_split_refuses_bad_seeds(svc):
    left, right = brick(svc, A_LEFT), brick(svc, A_RIGHT)
    assert not svc.split([], [right]).ok
    assert not svc.split([left], [brick(svc, B_SPOT)]).ok  # different objects
    assert not svc.split([left], [left]).ok  # same brick on both sides
    assert not svc.split([0], [right]).ok  # seed not on a cell
    assert count(svc) == 3
    assert svc.store.live_events() == []


def test_split_after_manual_merge_prefers_to_undo_the_manual_glue(svc):
    svc.merge(obj(svc, A_LEFT), obj(svc, B_SPOT), "alice")
    manual = [(a, b) for _, op, a, b in svc.store.live_events() if op == "add"]
    assert len(manual) == 1
    assert svc.split([brick(svc, A_LEFT)], [brick(svc, B_SPOT)], "bob").ok
    removed = [(a, b) for _, op, a, b in svc.store.live_events() if op == "remove"]
    assert removed == manual
    assert count(svc) == 3


def test_undo_merge_and_split(svc):
    start = svc.lut_copy()
    svc.merge(obj(svc, A_LEFT), obj(svc, C_SPOT), "alice")
    after_merge = svc.lut_copy()
    svc.split([brick(svc, A_LEFT)], [brick(svc, A_RIGHT)], "bob")
    result = svc.undo("alice")
    assert result.ok and "bob" in result.message
    assert (svc.lut_copy() == after_merge).all()
    assert svc.undo().ok
    assert (svc.lut_copy() == start).all()
    assert not svc.undo().ok


def test_restart_gives_the_same_picture(model, fingerprint, db_path):
    store, _ = EditStore.open(db_path, fingerprint)
    live = EditService(SupervoxelGraph(model), store)
    live.merge(obj(live, A_LEFT), obj(live, C_SPOT), "alice")
    live.split([brick(live, A_LEFT)], [brick(live, A_RIGHT)], "bob")
    live.merge(obj(live, B_SPOT), obj(live, A_RIGHT), "alice")
    expected = live.lut_copy()
    store.close()

    store2, status = EditStore.open(db_path, fingerprint)
    assert status == "match"
    restarted = EditService(SupervoxelGraph(model), store2)
    assert restarted.replayed_events == 3
    assert (restarted.lut_copy() == expected).all()


def test_undone_edits_stay_undone_after_restart(model, fingerprint, db_path):
    store, _ = EditStore.open(db_path, fingerprint)
    live = EditService(SupervoxelGraph(model), store)
    start = live.lut_copy()
    live.merge(obj(live, A_LEFT), obj(live, C_SPOT), "alice")
    live.undo()
    store.close()
    store2, _ = EditStore.open(db_path, fingerprint)
    assert (EditService(SupervoxelGraph(model), store2).lut_copy() == start).all()


def test_listeners_hear_about_each_edit(svc):
    heard = []
    svc.subscribe(heard.append)
    svc.merge(obj(svc, A_LEFT), obj(svc, C_SPOT), "alice")
    svc.undo()
    svc.merge(obj(svc, A_LEFT), obj(svc, A_LEFT))  # fails: no message to listeners
    assert len(heard) == 2 and all(r.changed for r in heard)
    assert svc.version == 2


def test_failed_save_changes_nothing(svc, monkeypatch):
    before = svc.lut_copy()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(svc.store, "append_edit", boom)
    try:
        svc.merge(obj(svc, A_LEFT), obj(svc, C_SPOT), "alice")
    except OSError:
        pass
    assert (svc.lut_copy() == before).all()
