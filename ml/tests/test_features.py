import numpy as np
import pytest

from poy_ml.features import N_FEATURES, landmarks_to_features, mirror_features


def hand(seed=0):
    return np.random.default_rng(seed).uniform(0, 1, size=(21, 3))


def test_shape_and_wrist_at_origin():
    f = landmarks_to_features(hand())
    assert f.shape == (N_FEATURES,)
    assert np.allclose(f[:3], 0)


def test_max_xy_distance_is_one():
    f = landmarks_to_features(hand()).reshape(21, 3)
    assert np.hypot(f[:, 0], f[:, 1]).max() == pytest.approx(1.0)


def test_translation_and_scale_invariant():
    h = hand(1)
    moved = h * 0.5 + np.array([0.2, 0.1, 0.3])
    assert np.allclose(landmarks_to_features(h), landmarks_to_features(moved))


def test_z_scaled_with_xy_not_by_itself():
    h = hand(2)
    h[5, 2] += 10  # a huge z must not change the scale
    f = landmarks_to_features(h).reshape(21, 3)
    assert np.hypot(f[:, 0], f[:, 1]).max() == pytest.approx(1.0)
    assert abs(f[5, 2]) > 1


def test_degenerate_hand_gives_zeros():
    assert not landmarks_to_features(np.full((21, 3), 0.5)).any()


def test_mirror_equals_flipping_the_image():
    h = hand(3)
    flipped = h.copy()
    flipped[:, 0] = 1 - flipped[:, 0]
    assert np.allclose(mirror_features(landmarks_to_features(h)), landmarks_to_features(flipped))


def test_mirror_works_on_batches_and_does_not_mutate():
    X = np.stack([landmarks_to_features(hand(i)) for i in range(3)])
    before = X.copy()
    M = mirror_features(X)
    assert np.array_equal(X, before)
    assert np.allclose(mirror_features(M), X)
