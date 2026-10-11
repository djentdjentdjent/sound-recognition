"""Load the saved classifier and predict environmental sounds from audio files."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sound_recognition.audio import AudioConfig, AudioLoadingError, load_audio
from sound_recognition.features import (
    IMPROVED_FEATURE_SCHEMA_VERSION,
    extract_improved_features,
    improved_feature_names,
)
from sound_recognition.model_training import (
    DEFAULT_MODEL_PATH,
    MODEL_ARTIFACT_VERSION,
)


class ModelArtifactError(ValueError):
    """Report a missing, damaged, or incompatible serialized model bundle."""


@dataclass(frozen=True)
class ClassPrediction:
    """One class returned in the ranked prediction list."""

    label: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """Top predictions and complete per-file processing time."""

    predictions: tuple[ClassPrediction, ...]
    processing_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation for the CLI and future API."""

        return {
            "predictions": [
                {"label": item.label, "confidence": item.confidence}
                for item in self.predictions
            ],
            "processing_ms": self.processing_ms,
        }


class AudioClassifier:
    """Reusable predictor that loads the estimator only once per process."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Load and validate all model components required for inference."""

        bundle_path = Path(model_path).expanduser()
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Model artifact does not exist: {bundle_path}")

        try:
            # Joblib artifacts must only be loaded from a trusted project source,
            # because deserialization can execute code contained in the artifact.
            bundle = joblib.load(bundle_path)
        except Exception as error:
            raise ModelArtifactError(
                f"Could not load model artifact: {bundle_path}"
            ) from error

        if not isinstance(bundle, dict):
            raise ModelArtifactError("Model artifact must contain a dictionary bundle.")

        required_keys = {
            "artifact_version",
            "audio_config",
            "classes",
            "estimator",
            "feature_names",
            "feature_schema_version",
        }
        missing_keys = required_keys.difference(bundle)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ModelArtifactError(f"Model artifact is missing keys: {missing}")

        if bundle["artifact_version"] != MODEL_ARTIFACT_VERSION:
            raise ModelArtifactError(
                "Unsupported model artifact version: "
                f"{bundle['artifact_version']} (expected {MODEL_ARTIFACT_VERSION})."
            )
        if bundle["feature_schema_version"] != IMPROVED_FEATURE_SCHEMA_VERSION:
            raise ModelArtifactError(
                "Model feature schema is incompatible with the code."
            )

        try:
            config = AudioConfig(**bundle["audio_config"])
        except (TypeError, ValueError) as error:
            raise ModelArtifactError("Model audio configuration is invalid.") from error

        expected_names = list(improved_feature_names(config))
        if list(bundle["feature_names"]) != expected_names:
            raise ModelArtifactError(
                "Model feature order is incompatible with the feature extractor."
            )

        estimator = bundle["estimator"]
        classes = tuple(str(label) for label in bundle["classes"])
        estimator_classes = tuple(
            str(label) for label in getattr(estimator, "classes_", ())
        )
        if not classes or classes != estimator_classes:
            raise ModelArtifactError(
                "Model class labels do not match the fitted estimator."
            )
        supports_decision_scores = callable(
            getattr(estimator, "decision_function", None)
        )
        supports_probabilities = callable(getattr(estimator, "predict_proba", None))
        if not supports_decision_scores and not supports_probabilities:
            raise ModelArtifactError(
                "Model estimator does not support confidence calculation."
            )

        self.model_path = bundle_path.resolve()
        self.config = config
        self.feature_names = tuple(expected_names)
        self.classes = classes
        self._estimator = estimator

    def predict(self, audio_path: str | Path, top_k: int = 3) -> PredictionResult:
        """Return the most probable classes for one supported audio file.

        Timing starts before decoding and includes preprocessing, feature
        extraction, confidence estimation, and ranking. Model loading is excluded
        because a server process creates this object once and reuses it for requests.
        """

        if not 1 <= top_k <= len(self.classes):
            raise ValueError(
                f"top_k must be between 1 and {len(self.classes)}, got {top_k}."
            )

        started = time.perf_counter()
        signal = load_audio(audio_path, self.config)
        features = extract_improved_features(signal, self.config)
        feature_row = features.reshape(1, -1)
        if callable(getattr(self._estimator, "decision_function", None)):
            decision_scores = np.asarray(
                self._estimator.decision_function(feature_row)[0],
                dtype=float,
            )
            # Subtracting the maximum prevents exponential overflow. Softmax
            # preserves score order and provides intuitive values that sum to one;
            # they are confidence scores, not independently calibrated probabilities.
            exponentials = np.exp(decision_scores - np.max(decision_scores))
            confidences = exponentials / np.sum(exponentials)
        else:
            confidences = np.asarray(
                self._estimator.predict_proba(feature_row)[0],
                dtype=float,
            )

        if (
            confidences.shape != (len(self.classes),)
            or not np.isfinite(confidences).all()
        ):
            raise RuntimeError("Model returned invalid class confidence values.")

        # Stable sorting makes equal-probability results reproducible by preserving
        # the estimator's class order while ranking from highest to lowest.
        ranked_indices = np.argsort(-confidences, kind="stable")[:top_k]
        predictions = tuple(
            ClassPrediction(
                label=self.classes[index],
                confidence=float(confidences[index]),
            )
            for index in ranked_indices
        )
        processing_ms = (time.perf_counter() - started) * 1_000
        return PredictionResult(predictions, processing_ms)


def parse_arguments() -> argparse.Namespace:
    """Parse a local audio path and optional prediction settings."""

    parser = argparse.ArgumentParser(
        description="Predict the three most probable ESC-50 sound classes."
    )
    parser.add_argument("audio", type=Path, help="Path to a WAV, FLAC, or OGG file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Print one prediction result as readable JSON."""

    arguments = parse_arguments()
    try:
        classifier = AudioClassifier(arguments.model)
        result = classifier.predict(arguments.audio, arguments.top_k)
    except (
        AudioLoadingError,
        FileNotFoundError,
        ModelArtifactError,
        ValueError,
    ) as error:
        raise SystemExit(f"Prediction failed: {error}") from error

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
