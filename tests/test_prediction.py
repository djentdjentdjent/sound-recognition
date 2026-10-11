"""Tests for loading a model bundle and predicting from supported audio files."""

from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.dummy import DummyClassifier

from sound_recognition.audio import AudioConfig, AudioLoadingError
from sound_recognition.features import (
    IMPROVED_FEATURE_SCHEMA_VERSION,
    improved_feature_names,
)
from sound_recognition.model_training import (
    DEFAULT_MODEL_PATH,
    MODEL_ARTIFACT_VERSION,
)
from sound_recognition.prediction import AudioClassifier, ModelArtifactError


def _write_test_model(path: Path, config: AudioConfig) -> None:
    """Create a small probability estimator with the production bundle schema."""

    names = list(improved_feature_names(config))
    estimator = DummyClassifier(strategy="prior")
    estimator.fit(
        np.zeros((4, len(names)), dtype=np.float32),
        np.asarray(["rain", "rain", "wind", "rain"]),
    )
    joblib.dump(
        {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "estimator": estimator,
            "audio_config": asdict(config),
            "feature_schema_version": IMPROVED_FEATURE_SCHEMA_VERSION,
            "feature_names": names,
            "classes": estimator.classes_.tolist(),
            "training": {},
        },
        path,
    )


def test_predict_returns_ranked_probabilities_and_complete_timing(
    tmp_path: Path,
) -> None:
    """A valid recording should produce ordered, finite prediction values."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    model_path = tmp_path / "model.joblib"
    audio_path = tmp_path / "tone.wav"
    _write_test_model(model_path, config)

    time_axis = np.arange(config.sample_rate) / config.sample_rate
    signal = (0.2 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
    sf.write(audio_path, signal, config.sample_rate)

    result = AudioClassifier(model_path).predict(audio_path, top_k=2)

    assert [item.label for item in result.predictions] == ["rain", "wind"]
    assert result.predictions[0].confidence >= result.predictions[1].confidence
    assert sum(item.confidence for item in result.predictions) == pytest.approx(1.0)
    assert result.processing_ms > 0.0


def test_predict_rejects_unsupported_audio_extension(tmp_path: Path) -> None:
    """Unsupported containers should fail before feature extraction starts."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    model_path = tmp_path / "model.joblib"
    text_path = tmp_path / "recording.txt"
    _write_test_model(model_path, config)
    text_path.write_text("not an audio stream", encoding="utf-8")

    with pytest.raises(AudioLoadingError, match="Unsupported audio file extension"):
        AudioClassifier(model_path).predict(text_path, top_k=2)


def test_predict_rejects_damaged_supported_file(tmp_path: Path) -> None:
    """A damaged WAV container should become a stable audio loading error."""

    config = AudioConfig(sample_rate=8_000, duration_seconds=1.0, n_fft=512)
    model_path = tmp_path / "model.joblib"
    damaged_path = tmp_path / "damaged.wav"
    _write_test_model(model_path, config)
    damaged_path.write_bytes(b"this is not a WAV file")

    with pytest.raises(AudioLoadingError, match="Could not decode audio file"):
        AudioClassifier(model_path).predict(damaged_path, top_k=2)


def test_model_loader_rejects_incompatible_bundle(tmp_path: Path) -> None:
    """A bundle with an unknown schema must never reach prediction code."""

    model_path = tmp_path / "incompatible.joblib"
    joblib.dump({"artifact_version": MODEL_ARTIFACT_VERSION}, model_path)

    with pytest.raises(ModelArtifactError, match="missing keys"):
        AudioClassifier(model_path)


def test_versioned_application_model_is_loadable() -> None:
    """The repository must include the trained model needed by the application."""

    classifier = AudioClassifier(DEFAULT_MODEL_PATH)

    assert len(classifier.classes) == 50
    assert len(classifier.feature_names) == 424
