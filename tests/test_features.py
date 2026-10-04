"""Tests for the fixed-length audio feature representation."""

import numpy as np

from sound_recognition.audio import AudioConfig
from sound_recognition.features import (
    extract_features,
    extract_improved_features,
    feature_names,
    improved_feature_names,
)


def test_extract_features_returns_named_finite_values() -> None:
    """Every configured recording must produce one finite value per feature name."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    time = np.arange(config.number_of_samples) / config.sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    values = extract_features(signal, config)

    assert values.shape == (len(feature_names(config)),)
    assert len(feature_names(config)) == 154
    assert np.isfinite(values).all()


def test_extract_improved_features_adds_finite_spectral_descriptors() -> None:
    """The richer representation should have stable names and finite values."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    time = np.arange(config.number_of_samples) / config.sample_rate
    signal = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    values = extract_improved_features(signal, config)

    assert values.shape == (len(improved_feature_names(config)),)
    assert len(improved_feature_names(config)) == 424
    assert np.isfinite(values).all()


def test_extract_improved_features_handles_silence() -> None:
    """Silent prepared audio must not create undefined spectral contrast."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    signal = np.zeros(config.number_of_samples, dtype=np.float32)

    values = extract_improved_features(signal, config)

    assert values.shape == (424,)
    assert np.isfinite(values).all()
