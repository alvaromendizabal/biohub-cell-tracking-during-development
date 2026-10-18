"""Numerical parity and invalid-input checks; no private data or fitted model."""
import numpy as np
import pytest

from biohub_tracking.fusion import weighted_harmonic_probability


@pytest.mark.parametrize("weight", [0.0, 0.1, 1.0, 5.0, 100.0])
def test_matches_original_reciprocal_form(weight):
    forward = np.array([0.0, 0.01, 0.2, 0.9, 1.0])
    reverse = np.array([0.8, 0.4, 0.2, 0.1, 1.0])
    expected = (1 + weight) / (1 / np.clip(forward, 1e-12, 1) + weight / np.clip(reverse, 1e-12, 1))
    actual = weighted_harmonic_probability(forward, reverse, weight=weight)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert np.isfinite(actual).all()


def test_identity_and_low_reverse_support():
    p = np.array([0.01, 0.5, 1.0])
    np.testing.assert_allclose(weighted_harmonic_probability(p, p, weight=0.5), p)
    assert weighted_harmonic_probability(np.array([0.9]), np.array([0.1]), weight=1)[0] < 0.5


@pytest.mark.parametrize("weight", [-1, np.nan, np.inf])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        weighted_harmonic_probability(np.array([0.5]), np.array([0.5]), weight=weight)


@pytest.mark.parametrize("forward,reverse", [([], []), ([0.5], [0.5, 0.5]), ([np.nan], [0.5]), ([1.01], [0.5]), ([0.5], [-0.1])])
def test_invalid_probabilities_rejected(forward, reverse):
    with pytest.raises(ValueError):
        weighted_harmonic_probability(np.array(forward), np.array(reverse), weight=0.5)


def test_large_finite_weight_stays_finite():
    actual = weighted_harmonic_probability(np.array([0.8]), np.array([0.2]), weight=1e300)
    np.testing.assert_allclose(actual, [0.2])
