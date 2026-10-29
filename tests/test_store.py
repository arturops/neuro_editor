from store import EditStore


def test_new_then_match(db_path):
    store, status = EditStore.open(db_path, {"k": 1})
    assert status == "new"
    store.append_edit("alice", [("add", 1, 2)])
    store.close()
    store, status = EditStore.open(db_path, {"k": 1})
    assert status == "match"
    assert store.live_events() == [(1, "add", 1, 2)]


def test_changed_fingerprint_moves_old_log_aside(db_path):
    store, _ = EditStore.open(db_path, {"k": 1})
    store.append_edit("alice", [("add", 1, 2)])
    store.close()
    store, status = EditStore.open(db_path, {"k": 2})
    assert status == "reset"
    assert store.live_events() == []
    backups = list(db_path.parent.glob("edits.sqlite.mismatch-*"))
    assert len(backups) == 1


def test_one_edit_can_hold_several_changes_and_undo_takes_them_all(db_path):
    store, _ = EditStore.open(db_path, {"k": 1})
    first = store.append_edit("alice", [("add", 1, 2)])
    second = store.append_edit("bob", [("remove", 3, 4), ("remove", 5, 6)])
    assert (first, second) == (1, 2)
    assert store.edit_count() == 2
    assert store.undo_last() == (2, "bob")
    assert store.live_events() == [(1, "add", 1, 2)]
    assert store.undo_last() == (1, "alice")
    assert store.undo_last() is None


def test_edit_ids_keep_growing_after_undo(db_path):
    store, _ = EditStore.open(db_path, {"k": 1})
    store.append_edit("a", [("add", 1, 2)])
    store.undo_last()
    assert store.append_edit("a", [("add", 1, 3)]) == 2
