import time
from types import SimpleNamespace

import numpy as np
import pytest

import fake_neuroglancer as fake
from conftest import A_LEFT, A_RIGHT, B_SPOT, C_SPOT
from viewer import KEYS, SEG_LAYER, ViewerApp


def make_app(svc, mode="per_user", tokens=("tok-one", "tok-two", "tok-three")):
    raw = np.zeros((6, 16, 40), dtype=np.uint8)
    app = ViewerApp(svc, raw, (40.0, 4.0, 4.0), "https://em.example.fly.dev/", 8080, tokens, mode, ng=fake)
    links = app.start()
    app.volumes_at_start = fake.LocalVolume.created
    return app, links


def reloads(app):
    """How many times the segmentation source has been swapped for a new one since start-up."""
    return fake.LocalVolume.created - app.volumes_at_start


def press(app, session, action, spot=None, selected=()):
    """Pretend a user pressed the key for `action` while the mouse was over voxel `spot` (z, y, x)."""
    z, y, x = spot if spot else (0, 0, 0)
    state = SimpleNamespace(
        viewer_state=SimpleNamespace(layers={SEG_LAYER: SimpleNamespace(segments=set(selected))}),
        mouse_voxel_coordinates=None if spot is None else np.array([x + 0.4, y + 0.4, z + 0.4]),
    )
    session.viewer.actions.handlers[action](state)


def message(session):
    return session.viewer.config_state.status_messages.get("em-msg", "")


def shown(app, spot):
    z, y, x = spot
    return int(app._display[x, y, z])  # Neuroglancer's array is (x, y, z)


def test_start_binds_all_interfaces_and_builds_public_links(svc):
    fake.bind_calls.clear()
    app, links = make_app(svc)
    assert fake.bind_calls == [("0.0.0.0", 8080)]
    assert links == [(f"user{i + 1}", f"https://em.example.fly.dev/v/{t}/") for i, t in enumerate(("tok-one", "tok-two", "tok-three"))]
    for session in app.sessions:
        assert set(session.viewer.config_state.input_event_bindings.viewer) == set(KEYS.values())
        assert set(session.viewer.actions.handlers) == set(KEYS)


def test_shared_mode_makes_one_viewer(svc):
    app, links = make_app(svc, mode="shared")
    assert len(app.sessions) == 1 and links[0][0] == "shared" and "/v/tok-one/" in links[0][1]


def test_layers_use_the_nanometre_voxel_size(svc):
    app, _ = make_app(svc)
    assert app.dims.names == ["x", "y", "z"] and app.dims.scales == [4.0, 4.0, 40.0] and app.dims.units == "nm"
    assert app._display.shape == (40, 16, 6)  # (x, y, z)


def test_merge_with_two_selected_objects_repaints_and_selects_result(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    a, c = svc.object_of(svc.brick_at(*A_LEFT)), svc.object_of(svc.brick_at(*C_SPOT))
    assert shown(app, A_LEFT) == a and shown(app, C_SPOT) == c
    press(app, one, "em-merge", selected={a, c})
    assert shown(app, A_LEFT) == shown(app, C_SPOT)
    assert reloads(app) == 1
    # every viewer now points at the new source, so browsers drop their cached 3D meshes
    assert all(s.viewer.state.layers[SEG_LAYER].source is app._seg_volume for s in app.sessions)
    assert one.viewer.state.layers[SEG_LAYER].segments == {shown(app, A_LEFT)}
    assert "Merged" in message(one)


def test_mesh_simplification_is_off_unless_asked_for(svc):
    app, _ = make_app(svc)
    assert app._seg_volume.mesh_options == {"max_quadrics_error": -1}
    press(app, app.sessions[0], "em-merge", selected={shown(app, A_LEFT), shown(app, C_SPOT)})
    assert app._seg_volume.mesh_options == {"max_quadrics_error": -1}  # also on the replacement source
    raw = np.zeros((6, 16, 40), dtype=np.uint8)
    on = ViewerApp(svc, raw, (40.0, 4.0, 4.0), "https://x.dev", 8080, ("t",), "shared", mesh_simplify=True, ng=fake)
    assert on._seg_volume.mesh_options is None  # Neuroglancer's own default


def test_merge_needs_exactly_two_selected(svc):
    app, _ = make_app(svc)
    press(app, app.sessions[0], "em-merge", selected={1})
    assert "exactly two" in message(app.sessions[0])
    assert reloads(app) == 0


def test_split_with_seeds_repaints_both_halves(svc):
    app, _ = make_app(svc)
    one, two = app.sessions[0], app.sessions[1]
    whole = shown(app, A_LEFT)
    two.viewer.state.layers[SEG_LAYER].segments = {whole}  # user 2 is looking at this object

    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-seed-b", A_RIGHT)
    assert "1 A and 1 B" in message(one)
    press(app, one, "em-split")

    left, right = shown(app, A_LEFT), shown(app, A_RIGHT)
    assert left != right
    assert one.viewer.state.layers[SEG_LAYER].segments == {left, right}  # editor sees both halves
    assert one.seeds == {"A": [], "B": []}
    # the other user keeps looking at "their" object: the half that holds its old number
    assert two.viewer.state.layers[SEG_LAYER].segments == {whole}
    assert reloads(app) == 1


def test_undo_restores_the_picture(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    start = app._display.copy()
    press(app, one, "em-merge", selected={shown(app, A_LEFT), shown(app, C_SPOT)})
    assert not (app._display == start).all()
    press(app, one, "em-undo")
    assert (app._display == start).all()
    assert reloads(app) == 2


def test_seed_on_empty_space_or_without_mouse_is_refused(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-seed-a", (3, 0, 0))  # y=0 is background in the made-up brain
    assert "No cell" in message(one) and one.seeds["A"] == []
    press(app, one, "em-seed-a", spot=None)
    assert "Hover" in message(one)


def test_split_without_seeds_explains_itself_and_changes_nothing(svc):
    app, _ = make_app(svc)
    press(app, app.sessions[0], "em-split")
    assert "at least one A seed" in message(app.sessions[0])
    assert reloads(app) == 0


def test_seeds_on_different_objects_warn(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-seed-b", B_SPOT)
    assert "different objects" in message(one)
    press(app, one, "em-clear-seeds")
    assert one.seeds == {"A": [], "B": []}


def test_remove_seed_removes_only_the_seed_under_the_mouse(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-seed-b", A_RIGHT)
    press(app, one, "em-remove-seed", A_RIGHT)
    assert len(one.seeds["B"]) == 0 and one.seed_points["B"] == []
    assert len(one.seeds["A"]) == 1 and len(one.seed_points["A"]) == 1
    assert "Removed seed B" in message(one)
    assert "seeds-B" not in one.viewer.state.layers and "seeds-A" in one.viewer.state.layers


def test_remove_seed_far_from_any_marker_or_without_mouse_changes_nothing(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-remove-seed", B_SPOT)
    assert "No seed under the mouse" in message(one) and len(one.seeds["A"]) == 1
    press(app, one, "em-remove-seed", spot=None)
    assert "Hover" in message(one) and len(one.seeds["A"]) == 1


def test_split_with_seeds_in_same_piece_mentions_how_to_fix_it(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-seed-b", A_LEFT)
    press(app, one, "em-split")
    assert "same small piece" in message(one)
    assert "Shift+R" in message(one) and "Shift+C" in message(one)


def test_second_seed_in_same_piece_is_refused_with_an_explanation(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-seed-a", A_LEFT)
    assert len(one.seeds["A"]) == 1 and len(one.seed_points["A"]) == 1
    assert "already placed" in message(one) and "Shift+R" in message(one)


def merged_app(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    start = svc.graph.object_count()
    press(app, one, "em-merge", selected={shown(app, A_LEFT), shown(app, C_SPOT)})
    assert svc.graph.object_count() == start - 1
    return app, one, start


def test_reset_all_needs_confirmation_then_resets_everyone(svc):
    app, one, start = merged_app(svc)
    two = app.sessions[1]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-reset-all")
    assert "Shift+Y" in message(one) and svc.graph.object_count() == start - 1  # only asked so far
    press(app, one, "em-confirm-yes")
    assert svc.graph.object_count() == start and svc.edit_count() == 0
    assert shown(app, A_LEFT) != shown(app, C_SPOT)  # the picture is repainted
    assert "Reset" in message(one) and one.seeds == {"A": [], "B": []}
    assert "user1 reset all edits" in message(two)


def test_reset_all_can_be_cancelled(svc):
    app, one, start = merged_app(svc)
    press(app, one, "em-reset-all")
    press(app, one, "em-confirm-no")
    assert "cancelled" in message(one) and svc.graph.object_count() == start - 1
    press(app, one, "em-confirm-yes")  # the question is gone, so yes does nothing
    assert "Nothing to confirm" in message(one) and svc.graph.object_count() == start - 1


def test_reset_all_confirmation_expires_and_needs_edits(svc):
    app, _ = make_app(svc)
    one = app.sessions[0]
    press(app, one, "em-reset-all")
    assert "No edits to reset" in message(one)
    app, one, start = merged_app(svc)
    press(app, one, "em-reset-all")
    one.reset_deadline = time.monotonic() - 1  # the 15 seconds are over
    press(app, one, "em-confirm-yes")
    assert "Nothing to confirm" in message(one) and svc.graph.object_count() == start - 1


def test_seeds_are_private_to_each_user_and_survive_other_users_edits(svc):
    app, _ = make_app(svc)
    one, two = app.sessions[0], app.sessions[1]
    press(app, one, "em-seed-a", A_LEFT)
    press(app, one, "em-seed-b", A_RIGHT)
    # user2 edits the shared data: user1's seeds and markers stay put
    press(app, two, "em-merge", selected={shown(app, A_LEFT), shown(app, C_SPOT)})
    assert len(one.seeds["A"]) == 1 and len(one.seeds["B"]) == 1
    assert "seeds-A" in one.viewer.state.layers and "seeds-B" in one.viewer.state.layers
    # user2 has no seeds of their own, so user1's seeds do not count for user2's split
    assert two.seeds == {"A": [], "B": []} and "seeds-A" not in two.viewer.state.layers
    press(app, two, "em-split")
    assert "at least one A seed" in message(two)


def test_edit_log_uses_user_labels_not_secret_tokens(svc):
    app, _ = make_app(svc)
    press(app, app.sessions[1], "em-merge", selected={shown(app, A_LEFT), shown(app, C_SPOT)})
    users = {row[0] for row in svc.store._conn.execute("SELECT DISTINCT user FROM edge_events")}
    assert users == {"user2"}


def test_a_crashing_action_is_reported_not_raised(svc, monkeypatch):
    app, _ = make_app(svc)

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(svc, "merge", boom)
    press(app, app.sessions[0], "em-merge", selected={shown(app, A_LEFT), shown(app, C_SPOT)})
    assert "kaboom" in message(app.sessions[0])
