"""Reproducible baseline experiments for the ESC-50 dataset."""

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sound_recognition.audio import DEFAULT_AUDIO_CONFIG, AudioConfig
from sound_recognition.dataset import extract_dataset_features, load_esc50_metadata
from sound_recognition.features import FEATURE_SCHEMA_VERSION, feature_names

RANDOM_STATE = 42
TEST_FOLD = 5


def build_models() -> Mapping[str, ClassifierMixin]:
    """Create all estimators with fixed parameters for reproducible comparison."""

    return {
        "Most frequent class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3_000, random_state=RANDOM_STATE),
        ),
        "RBF support vector machine": make_pipeline(
            StandardScaler(),
            SVC(C=10.0, kernel="rbf", gamma="scale"),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _config_signature(config: AudioConfig) -> str:
    """Serialize preprocessing parameters for safe feature-cache validation."""

    signature = {
        "audio_config": asdict(config),
        # Incrementing this value invalidates caches when feature definitions
        # change even if all audio preprocessing parameters remain identical.
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    return json.dumps(signature, sort_keys=True)


def load_or_extract_features(
    metadata: pd.DataFrame,
    cache_path: str | Path,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
    force_extraction: bool = False,
) -> NDArray[np.float32]:
    """Load a matching local feature cache or extract all ESC-50 recordings."""

    cache = Path(cache_path)
    expected_signature = _config_signature(config)
    expected_filenames = metadata["filename"].to_numpy(dtype=str)

    if cache.is_file() and not force_extraction:
        with np.load(cache, allow_pickle=False) as stored:
            cache_is_compatible = (
                str(stored["config_signature"].item()) == expected_signature
                and np.array_equal(stored["filenames"], expected_filenames)
                and np.array_equal(
                    stored["feature_names"],
                    np.asarray(feature_names(config)),
                )
            )
            if cache_is_compatible:
                print(f"Loading cached features from {cache}")
                return np.asarray(stored["features"], dtype=np.float32)

        print(
            "The feature cache does not match the current configuration; rebuilding it."
        )

    def print_progress(completed: int, total: int) -> None:
        # Updating every 100 files gives useful feedback without printing 2,000 lines.
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(f"Extracted features: {completed}/{total}")

    features = extract_dataset_features(metadata, config, print_progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        features=features,
        filenames=expected_filenames,
        feature_names=np.asarray(feature_names(config)),
        config_signature=np.asarray(expected_signature),
    )
    print(f"Saved feature cache to {cache}")
    return features


def evaluate_models(
    features: NDArray[np.floating],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, str, NDArray[np.str_], NDArray[np.str_]]:
    """Fit every model on folds 1-4 and evaluate it once on official fold 5."""

    train_mask = metadata["fold"].to_numpy() != TEST_FOLD
    test_mask = ~train_mask
    labels = metadata["category"].to_numpy(dtype=str)
    train_features = features[train_mask]
    test_features = features[test_mask]
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

    rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, NDArray[np.str_]] = {}

    for model_name, model in build_models().items():
        print(f"Training: {model_name}")
        fit_started = time.perf_counter()
        model.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - fit_started

        prediction_started = time.perf_counter()
        predictions = np.asarray(model.predict(test_features), dtype=str)
        prediction_seconds = time.perf_counter() - prediction_started
        predictions_by_model[model_name] = predictions

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(test_labels, predictions),
                "macro_f1": f1_score(
                    test_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "fit_seconds": fit_seconds,
                "prediction_ms_per_recording": (
                    prediction_seconds / len(test_features) * 1_000
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
        ignore_index=True,
    )
    best_model_name = str(results.iloc[0]["model"])
    return results, best_model_name, test_labels, predictions_by_model[best_model_name]


def save_confusion_matrix(
    true_labels: NDArray[np.str_],
    predicted_labels: NDArray[np.str_],
    model_name: str,
    output_path: str | Path,
) -> None:
    """Save a row-normalized confusion matrix for the best tested model."""

    class_names = np.asarray(sorted(np.unique(true_labels)))
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_names,
        normalize="true",
    )

    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Recall")
    axis.set(
        title=f"Normalized confusion matrix: {model_name}",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
    )
    axis.set_xticklabels(class_names, rotation=90, fontsize=7)
    axis.set_yticklabels(class_names, fontsize=7)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def save_largest_confusions(
    true_labels: NDArray[np.str_],
    predicted_labels: NDArray[np.str_],
    output_path: str | Path,
) -> None:
    """Save the largest off-diagonal errors as a compact, readable CSV table."""

    class_names = np.asarray(sorted(np.unique(true_labels)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    rows: list[dict[str, Any]] = []
    for true_index, predicted_index in zip(*np.nonzero(matrix), strict=True):
        if true_index != predicted_index:
            rows.append(
                {
                    "true_class": class_names[true_index],
                    "predicted_class": class_names[predicted_index],
                    "count": int(matrix[true_index, predicted_index]),
                }
            )

    confusions = pd.DataFrame(rows).sort_values(
        ["count", "true_class", "predicted_class"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    confusions.to_csv(destination, index=False)


def run_experiment(
    dataset_directory: str | Path,
    output_directory: str | Path,
    cache_path: str | Path,
    force_features: bool = False,
) -> pd.DataFrame:
    """Run feature extraction, model comparison, and result generation."""

    config = AudioConfig()
    metadata = load_esc50_metadata(dataset_directory)
    features = load_or_extract_features(
        metadata,
        cache_path,
        config,
        force_extraction=force_features,
    )
    results, best_model_name, true_labels, predicted_labels = evaluate_models(
        features,
        metadata,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "baseline_metrics.csv", index=False, float_format="%.6f")
    save_confusion_matrix(
        true_labels,
        predicted_labels,
        best_model_name,
        output / "confusion_matrix.png",
    )
    save_largest_confusions(
        true_labels,
        predicted_labels,
        output / "largest_confusions.csv",
    )

    print("\nModel comparison on ESC-50 fold 5:")
    print(results.to_string(index=False))
    print(f"\nMost promising model: {best_model_name}")
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse paths and cache options for the command-line experiment."""

    parser = argparse.ArgumentParser(
        description="Extract ESC-50 features and compare classical classifiers."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache/esc50_features.npz"),
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore an existing cache and extract all audio features again.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ESC-50 experiment from the command line."""

    arguments = parse_arguments()
    run_experiment(
        arguments.dataset,
        arguments.output,
        arguments.cache,
        force_features=arguments.force_features,
    )


if __name__ == "__main__":
    main()
