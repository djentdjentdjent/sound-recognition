"""Tests for the public FastAPI endpoints and upload validation."""

from pathlib import Path

from fastapi.testclient import TestClient

from sound_recognition.api import create_app
from sound_recognition.audio import AudioLoadingError
from sound_recognition.prediction import ClassPrediction, PredictionResult


class StubClassifier:
    """Return deterministic predictions without running feature extraction."""

    classes = ("dog", "rain", "wind")

    def __init__(self, *, reject_audio: bool = False) -> None:
        self.reject_audio = reject_audio
        self.received_path: Path | None = None

    def predict(self, audio_path: str | Path, top_k: int = 3) -> PredictionResult:
        """Record the temporary path and imitate the production predictor."""

        self.received_path = Path(audio_path)
        assert self.received_path.is_file()
        assert self.received_path.suffix == ".wav"
        assert top_k == 3

        if self.reject_audio:
            raise AudioLoadingError("Could not decode uploaded audio.")

        return PredictionResult(
            predictions=(
                ClassPrediction(label="rain", confidence=0.8),
                ClassPrediction(label="wind", confidence=0.15),
                ClassPrediction(label="dog", confidence=0.05),
            ),
            processing_ms=12.5,
        )


def test_health_and_classes_describe_loaded_classifier() -> None:
    """Readiness and class metadata should reflect the startup classifier."""

    with TestClient(create_app(classifier=StubClassifier())) as client:
        health_response = client.get("/health")
        classes_response = client.get("/classes")

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "model_loaded": True,
        "classes_count": 3,
    }
    assert classes_response.status_code == 200
    assert classes_response.json() == {
        "count": 3,
        "classes": ["dog", "rain", "wind"],
    }


def test_predict_returns_top_three_and_removes_temporary_file() -> None:
    """A valid upload should be classified and deleted after processing."""

    classifier = StubClassifier()
    with TestClient(create_app(classifier=classifier)) as client:
        response = client.post(
            "/predict",
            files={"file": ("rain.wav", b"small audio body", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "rain.wav",
        "predictions": [
            {"label": "rain", "confidence": 0.8},
            {"label": "wind", "confidence": 0.15},
            {"label": "dog", "confidence": 0.05},
        ],
        "processing_ms": 12.5,
    }
    assert classifier.received_path is not None
    assert not classifier.received_path.exists()


def test_predict_rejects_unsupported_extension() -> None:
    """Formats outside the documented WAV, FLAC, and OGG set return HTTP 415."""

    with TestClient(create_app(classifier=StubClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("recording.mp3", b"audio", "audio/mpeg")},
        )

    assert response.status_code == 415
    assert "Supported extensions" in response.json()["detail"]


def test_predict_rejects_media_type_that_does_not_match_extension() -> None:
    """A misleading declared media type should fail before model execution."""

    with TestClient(create_app(classifier=StubClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("recording.wav", b"audio", "text/plain")},
        )

    assert response.status_code == 415
    assert "does not match" in response.json()["detail"]


def test_predict_rejects_file_larger_than_configured_limit() -> None:
    """Streaming validation should return HTTP 413 once the limit is crossed."""

    with TestClient(
        create_app(classifier=StubClassifier(), max_upload_bytes=4)
    ) as client:
        response = client.post(
            "/predict",
            files={"file": ("recording.wav", b"12345", "audio/wav")},
        )

    assert response.status_code == 413
    assert "Maximum size is 4 bytes" in response.json()["detail"]


def test_predict_rejects_empty_file() -> None:
    """An empty upload should return a clear validation error."""

    with TestClient(create_app(classifier=StubClassifier())) as client:
        response = client.post(
            "/predict",
            files={"file": ("recording.wav", b"", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Uploaded file is empty."


def test_predict_turns_decoder_failure_into_client_error() -> None:
    """Damaged supported containers should return HTTP 422 without internals."""

    with TestClient(create_app(classifier=StubClassifier(reject_audio=True))) as client:
        response = client.post(
            "/predict",
            files={"file": ("damaged.wav", b"not a wav stream", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Could not decode uploaded audio."
