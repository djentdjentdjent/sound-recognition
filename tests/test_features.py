"""Tests for the fixed-length audio feature representation."""

import numpy as np

from sound_recognition.audio import AudioConfig
from sound_recognition.features import extract_features, feature_names


def test_extract_features_returns_named_finite_values() -> None:
    """Every configured recording must produce one finite value per feature name."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    time = np.arange(config.number_of_samples) / config.sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    values = extract_features(signal, config)

    assert values.shape == (len(feature_names(config)),)
    assert len(feature_names(config)) == 154
    assert np.isfinite(values).all()
