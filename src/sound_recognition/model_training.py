"""Train and serialize the selected environmental sound classifier."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sound_recognition.audio import AudioConfig, load_audio
from sound_recognition.dataset import load_esc50_metadata
from sound_recognition.experiments import TEST_FOLD
from sound_recognition.features import (
    IMPROVED_FEATURE_SCHEMA_VERSION,
    extract_improved_features,
    improved_feature_names,
)
from sound_recognition.improved_experiments import load_or_extract_improved_features

MODEL_ARTIFACT_VERSION = 1
DEFAULT_MODEL_PATH = Path("artifacts/model/sound_classifier.joblib")
DEFAULT_METADATA_PATH = Path("artifacts/model/model_metadata.json")


def build_selected_classifier() -> Pipeline:
    """Create the exact classifier configuration selected by cross-validation.

    The model keeps the same decision function that produced the recorded test
    metrics. The inference module converts its multiclass scores to normalized
    confidence values without changing the fitted decision boundary.
    """

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                ),
            ),
        ]
    )


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalar values into ordinary values accepted by JSON."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def benchmark_end_to_end_inference(
    estimator: Pipeline,
    audio_paths: list[Path],
    config: AudioConfig,
) -> dict[str, Any]:
    """Measure decoding, preprocessing, feature extraction, and prediction."""

    if not audio_paths:
        raise ValueError("At least one audio path is required for benchmarking.")

    # The warm-up absorbs one-time library initialization so the summary better
    # represents requests handled by a long-running application process.
    warm_up_signal = load_audio(audio_paths[0], config)
    warm_up_features = extract_improved_features(warm_up_signal, config)
    estimator.decision_function(warm_up_features.reshape(1, -1))

    durations_ms: list[float] = []
    for audio_path in audio_paths:
        started = time.perf_counter()
        signal = load_audio(audio_path, config)
        features = extract_improved_features(signal, config)
        estimator.decision_function(features.reshape(1, -1))
        durations_ms.append((time.perf_counter() - started) * 1_000)

    durations = np.asarray(durations_ms)
    return {
        "recordings": len(audio_paths),
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "min_ms": float(np.min(durations)),
        "max_ms": float(np.max(durations)),
        "includes": [
            "audio decoding",
            "resampling and normalization",
            "feature extraction",
            "confidence estimation",
        ],
        "excludes": ["model loading"],
    }


def train_and_save_model(
    dataset_directory: str | Path,
    feature_cache_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    force_features: bool = False,
    benchmark_recordings: int = 50,
) -> dict[str, Any]:
    """Fit the selected model on folds 1-4 and save a self-describing bundle."""

    config = AudioConfig()
    metadata = load_esc50_metadata(dataset_directory)
    features = load_or_extract_improved_features(
        metadata,
        feature_cache_path,
        config,
        force_extraction=force_features,
    )
    labels = metadata["category"].to_numpy(dtype=str)
    folds = metadata["fold"].to_numpy(dtype=int)
    training_mask = folds != TEST_FOLD
    test_mask = ~training_mask

    estimator = build_selected_classifier()
    fit_started = time.perf_counter()
    estimator.fit(features[training_mask], labels[training_mask])
    fit_seconds = time.perf_counter() - fit_started

    # Fold 5 remains excluded from fitting. Evaluating it here verifies that the
    # serialized application estimator preserves the previously selected quality.
    prediction_started = time.perf_counter()
    test_predictions = estimator.predict(features[test_mask])
    cached_prediction_seconds = time.perf_counter() - prediction_started
    evaluation = {
        "accuracy": float(accuracy_score(labels[test_mask], test_predictions)),
        "macro_f1": float(
            f1_score(
                labels[test_mask],
                test_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "cached_prediction_ms_per_recording": float(
            cached_prediction_seconds / int(test_mask.sum()) * 1_000
        ),
    }

    if benchmark_recordings <= 0:
        raise ValueError("benchmark_recordings must be a positive integer.")
    test_audio_paths = [
        Path(path)
        for path in metadata.loc[test_mask, "audio_path"].iloc[:benchmark_recordings]
    ]
    inference_benchmark = benchmark_end_to_end_inference(
        estimator,
        test_audio_paths,
        config,
    )

    names = list(improved_feature_names(config))
    classes = estimator.classes_.tolist()
    model_bundle = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "estimator": estimator,
        "audio_config": asdict(config),
        "feature_schema_version": IMPROVED_FEATURE_SCHEMA_VERSION,
        "feature_names": names,
        "classes": classes,
        "training": {
            "dataset": "ESC-50",
            "training_folds": sorted(np.unique(folds[training_mask]).tolist()),
            "test_fold": TEST_FOLD,
            "training_recordings": int(training_mask.sum()),
            "model": "RBF support vector machine",
            "parameters": {
                "C": 10.0,
                "gamma": "scale",
                "kernel": "rbf",
                "confidence_method": "softmax of one-vs-rest decision scores",
            },
            "fit_seconds": fit_seconds,
            "evaluation": evaluation,
            "end_to_end_inference_benchmark": inference_benchmark,
        },
    }

    output_model = Path(model_path)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    # Compression keeps the versioned binary small while preserving every part
    # of the fitted pipeline, including the scaler and support vectors.
    joblib.dump(model_bundle, output_model, compress=3)

    package_versions = {
        package: version(package)
        for package in ("joblib", "librosa", "numpy", "scikit-learn", "soundfile")
    }
    readable_metadata = {
        key: value for key, value in model_bundle.items() if key != "estimator"
    }
    readable_metadata["python_version"] = platform.python_version()
    readable_metadata["package_versions"] = package_versions
    readable_metadata["model_file"] = output_model.name
    readable_metadata["model_size_bytes"] = output_model.stat().st_size

    output_metadata = Path(metadata_path)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(readable_metadata, indent=2, default=_json_ready) + "\n",
        encoding="utf-8",
    )
    return readable_metadata


def parse_arguments() -> argparse.Namespace:
    """Parse paths used to reproduce the application model artifact."""

    parser = argparse.ArgumentParser(
        description="Train and serialize the selected ESC-50 application model."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/ESC-50"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("artifacts/cache/esc50_improved_features.npz"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--benchmark-recordings",
        type=int,
        default=50,
        help="Number of untouched fold 5 files used for end-to-end timing.",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore a compatible cache and extract improved features again.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the application model and display its saved evaluation summary."""

    arguments = parse_arguments()
    metadata = train_and_save_model(
        arguments.dataset,
        arguments.feature_cache,
        arguments.model,
        arguments.metadata,
        force_features=arguments.force_features,
        benchmark_recordings=arguments.benchmark_recordings,
    )
    evaluation = metadata["training"]["evaluation"]
    print(f"Saved model to {arguments.model}")
    print(f"Saved metadata to {arguments.metadata}")
    print(f"Fold 5 accuracy: {evaluation['accuracy']:.4f}")
    print(f"Fold 5 macro F1: {evaluation['macro_f1']:.4f}")
    benchmark = metadata["training"]["end_to_end_inference_benchmark"]
    print(
        "Mean end-to-end inference time: "
        f"{benchmark['mean_ms']:.2f} ms over {benchmark['recordings']} recordings"
    )


if __name__ == "__main__":
    main()
