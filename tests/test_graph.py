import numpy as np

from conftest import A_LEFT, A_MID_LEFT, A_MID_RIGHT, B_SPOT, BLOCK, C_SPOT, CROP, make_labels
from graph import make_pair, make_fingerprint


def test_every_neuron_voxel_has_a_brick_and_each_brick_has_one_neuron(model):
    cleaned_nonzero = model.bricks > 0
    labels_foreground = make_labels() != np.uint64(2**64 - 1)
    assert (cleaned_nonzero == labels_foreground).all()
    for brick in range(1, model.n_bricks + 1):
        neurons = np.unique(model.neuron_of[brick])
        assert neurons.size == 1


def test_starting_objects_are_the_three_neurons(graph):
    assert graph.object_count() == 3
    a = graph.object_of(graph.brick_at(*A_LEFT))
    assert a == graph.object_of(graph.brick_at(*A_MID_LEFT)) == graph.object_of(graph.brick_at(3, 5, 35))
    assert len({a, graph.object_of(graph.brick_at(*B_SPOT)), graph.object_of(graph.brick_at(*C_SPOT))}) == 3
    assert graph.lut[0] == 0


def test_object_number_is_smallest_brick_in_group(graph):
    for obj in graph.object_ids():
        assert obj == graph.members(obj).min()


def test_glue_thickness_uses_real_face_sizes(graph):
    left = graph.brick_at(*A_MID_LEFT)
    far_left = graph.brick_at(*A_LEFT)
    right = graph.brick_at(*A_MID_RIGHT)
    # x-direction faces are y*z = 4 nm * 40 nm = 160 nm^2 per voxel face
    assert graph.edges[make_pair(far_left, left)] == 7 * 6 * 160  # full cross-section of the thick bar
    assert graph.edges[make_pair(left, right)] == 2 * 2 * 160  # only the neck


def test_touching_between_different_neurons_is_recorded_but_is_not_glue(graph):
    a = graph.brick_at(3, 8, 2)
    c = graph.brick_at(*C_SPOT)
    assert make_pair(a, c) in graph._contact
    assert make_pair(a, c) not in graph.edges


def test_fingerprint_changes_when_bricks_change(model):
    base = make_fingerprint(model, CROP[0], CROP[1], BLOCK)
    assert base == make_fingerprint(model, CROP[0], CROP[1], BLOCK)
    assert base != make_fingerprint(model, (1, 0, 0), CROP[1], BLOCK)
    changed = make_fingerprint(model, CROP[0], CROP[1], (3, 16, 10))
    assert base != changed


def test_reset_goes_back_to_start(graph):
    before = graph.lut.copy()
    graph.add_edge(graph.brick_at(*A_LEFT), graph.brick_at(*B_SPOT))
    graph.recompute()
    assert graph.object_count() == 2
    graph.reset()
    assert (graph.lut == before).all()
