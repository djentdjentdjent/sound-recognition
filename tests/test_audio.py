"""Tests for deterministic audio preprocessing."""

import numpy as np
import pytest

from sound_recognition.audio import AudioConfig, prepare_audio


def test_prepare_audio_downmixes_resamples_resizes_and_normalizes() -> None:
    """A stereo signal should become a normalized fixed-size mono signal."""

    source_sample_rate = 8_000
    source_time = np.arange(source_sample_rate // 2) / source_sample_rate
    stereo_signal = np.column_stack(
        (
            0.2 * np.sin(2 * np.pi * 220 * source_time),
            0.4 * np.sin(2 * np.pi * 440 * source_time),
        )
    )
    config = AudioConfig(sample_rate=16_000, duration_seconds=1.0)

    prepared = prepare_audio(stereo_signal, source_sample_rate, config)

    assert prepared.shape == (16_000,)
    assert prepared.dtype == np.float32
    assert np.max(np.abs(prepared)) == pytest.approx(config.peak_amplitude)
    # The original recording lasts half a second, so the second half is padding.
    assert np.allclose(prepared[8_100:], 0.0, atol=1e-5)


def test_prepare_audio_preserves_silence() -> None:
    """Peak normalization must not divide a silent signal by zero."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0)
    prepared = prepare_audio(np.zeros(2_000, dtype=np.float32), 8_000, config)

    assert prepared.shape == (8_000,)
    assert np.isfinite(prepared).all()
    assert np.count_nonzero(prepared) == 0
