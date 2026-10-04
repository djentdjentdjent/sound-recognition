"""Model selection and final evaluation for the improved ESC-50 classifier."""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sound_recognition.audio import DEFAULT_AUDIO_CONFIG, AudioConfig
from sound_recognition.dataset import (
    extract_dataset_improved_features,
    load_esc50_metadata,
)
from sound_recognition.experiments import (
    RANDOM_STATE,
    TEST_FOLD,
    load_or_extract_features,
    save_confusion_matrix,
    save_largest_confusions,
)
from sound_recognition.features import (
    IMPROVED_FEATURE_SCHEMA_VERSION,
    improved_feature_names,
)

BASELINE_MODEL_NAME = "Baseline RBF support vector machine"


def _improved_config_signature(config: AudioConfig) -> str:
    """Serialize everything that changes the improved cached feature matrix."""

    signature = {
        "audio_config": asdict(config),
        "improved_feature_schema_version": IMPROVED_FEATURE_SCHEMA_VERSION,
    }
    return json.dumps(signature, sort_keys=True)


def load_or_extract_improved_features(
    metadata: pd.DataFrame,
    cache_path: str | Path,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
    force_extraction: bool = False,
) -> NDArray[np.float32]:
    """Load a compatible improved-feature cache or rebuild it from WAV files."""

    cache = Path(cache_path)
    expected_signature = _improved_config_signature(config)
    expected_filenames = metadata["filename"].to_numpy(dtype=str)
    expected_feature_names = np.asarray(improved_feature_names(config))

    if cache.is_file() and not force_extraction:
        with np.load(cache, allow_pickle=False) as stored:
            cache_is_compatible = (
                str(stored["config_signature"].item()) == expected_signature
                and np.array_equal(stored["filenames"], expected_filenames)
                and np.array_equal(stored["feature_names"], expected_feature_names)
            )
            if cache_is_compatible:
                print(f"Loading cached improved features from {cache}")
                return np.asarray(stored["features"], dtype=np.float32)

        print("The improved feature cache is incompatible; rebuilding it.")

    def print_progress(completed: int, total: int) -> None:
        # Periodic progress is useful because extracting 2,000 recordings takes
        # substantially longer than fitting any individual classical model.
        if completed == 1 or completed % 100 == 0 or completed == total:
            print(f"Extracted improved features: {completed}/{total}")

    features = extract_dataset_improved_features(metadata, config, print_progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        features=features,
        filenames=expected_filenames,
        feature_names=expected_feature_names,
        config_signature=np.asarray(expected_signature),
    )
    print(f"Saved improved feature cache to {cache}")
    return features


def build_searches() -> Mapping[
    str, tuple[ClassifierMixin, Mapping[str, Sequence[Any]]]
]:
    """Create compact, interpretable grids for the two strongest baseline models."""

    svm = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", SVC(kernel="rbf")),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    return {
        "RBF support vector machine": (
            svm,
            {
                "classifier__C": (1.0, 10.0, 30.0, 100.0),
                "classifier__gamma": ("scale", "auto", 0.001, 0.01),
            },
        ),
        "Random forest": (
            forest,
            {
                "max_features": ("sqrt", 0.5),
                "min_samples_leaf": (1, 2),
            },
        ),
    }


def _training_fold_split(folds: NDArray[np.integer]) -> PredefinedSplit:
    """Map original folds 1-4 to four validation splits without fold 5."""

    if np.any(folds == TEST_FOLD):
        raise ValueError("The model-selection data must not contain test fold 5.")
    unique_folds = sorted(np.unique(folds).tolist())
    fold_to_split = {fold: index for index, fold in enumerate(unique_folds)}
    return PredefinedSplit(np.asarray([fold_to_split[fold] for fold in folds]))


def tune_models(
    feature_sets: Mapping[str, NDArray[np.floating]],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, str, ClassifierMixin, str]:
    """Select a representation and estimator using only training folds 1-4."""

    train_mask = metadata["fold"].to_numpy() != TEST_FOLD
    training_labels = metadata.loc[train_mask, "category"].to_numpy(dtype=str)
    training_folds = metadata.loc[train_mask, "fold"].to_numpy(dtype=int)
    cross_validation = _training_fold_split(training_folds)

    all_rows: list[pd.DataFrame] = []
    fitted_searches: dict[tuple[str, str], GridSearchCV] = {}
    for representation_name, features in feature_sets.items():
        training_features = features[train_mask]
        for model_name, (estimator, parameter_grid) in build_searches().items():
            print(f"Tuning {model_name} with {representation_name} features")
            search = GridSearchCV(
                estimator,
                parameter_grid,
                scoring="f1_macro",
                cv=cross_validation,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )
            search.fit(training_features, training_labels)
            fitted_searches[(representation_name, model_name)] = search

            rows = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ].copy()
            rows.insert(0, "model", model_name)
            rows.insert(0, "representation", representation_name)
            rows["parameters"] = rows.pop("params").map(
                lambda value: json.dumps(value, sort_keys=True)
            )
            all_rows.append(rows)

    tuning_results = pd.concat(all_rows, ignore_index=True).sort_values(
        ["mean_test_score", "std_test_score"],
        ascending=[False, True],
        ignore_index=True,
    )
    best_row = tuning_results.iloc[0]
    best_representation = str(best_row["representation"])
    best_model_name = str(best_row["model"])
    best_search = fitted_searches[(best_representation, best_model_name)]
    return (
        tuning_results,
        best_representation,
        best_search.best_estimator_,
        best_model_name,
    )


def evaluate_selected_model(
    estimator: ClassifierMixin,
    features: NDArray[np.floating],
    metadata: pd.DataFrame,
) -> tuple[dict[str, Any], NDArray[np.str_], NDArray[np.str_]]:
    """Refit the selected configuration and evaluate fold 5 exactly once."""

    train_mask = metadata["fold"].to_numpy() != TEST_FOLD
    test_mask = ~train_mask
    labels = metadata["category"].to_numpy(dtype=str)

    fit_started = time.perf_counter()
    estimator.fit(features[train_mask], labels[train_mask])
    fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    predictions = np.asarray(estimator.predict(features[test_mask]), dtype=str)
    prediction_seconds = time.perf_counter() - prediction_started
    true_labels = labels[test_mask]

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "fit_seconds": fit_seconds,
        "prediction_ms_per_recording": prediction_seconds / len(predictions) * 1_000,
    }
    return metrics, true_labels, predictions


def save_per_class_metrics(
    true_labels: NDArray[np.str_],
    predicted_labels: NDArray[np.str_],
    output_path: str | Path,
) -> None:
    """Save precision, recall, and F1 for every class for error analysis."""

    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )
    class_names = sorted(np.unique(true_labels).tolist())
    rows = [{"class": name, **report[name]} for name in class_names]
    pd.DataFrame(rows).sort_values("f1-score").to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )


def run_experiment(
    dataset_directory: str | Path,
    output_directory: str | Path,
    baseline_cache_path: str | Path,
    improved_cache_path: str | Path,
    force_features: bool = False,
) -> pd.DataFrame:
    """Tune candidate models, evaluate the winner, and write all artifacts."""

    config = AudioConfig()
    metadata = load_esc50_metadata(dataset_directory)
    baseline_features = load_or_extract_features(
        metadata,
        baseline_cache_path,
        config,
        force_extraction=force_features,
    )
    improved_features = load_or_extract_improved_features(
        metadata,
        improved_cache_path,
        config,
        force_extraction=force_features,
    )
    feature_sets = {
        "baseline": baseline_features,
        "improved": improved_features,
    }

    tuning_results, representation, estimator, model_name = tune_models(
        feature_sets,
        metadata,
    )
    final_metrics, true_labels, predicted_labels = evaluate_selected_model(
        estimator,
        feature_sets[representation],
        metadata,
    )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(
        output / "tuning_results.csv",
        index=False,
        float_format="%.6f",
    )

    baseline_metrics = pd.read_csv(output / "baseline_metrics.csv")
    baseline_svm = baseline_metrics.loc[
        baseline_metrics["model"] == "RBF support vector machine"
    ].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "model": BASELINE_MODEL_NAME,
                "representation": "baseline",
                "feature_count": baseline_features.shape[1],
                "accuracy": baseline_svm["accuracy"],
                "macro_f1": baseline_svm["macro_f1"],
                "fit_seconds": baseline_svm["fit_seconds"],
                "prediction_ms_per_recording": baseline_svm[
                    "prediction_ms_per_recording"
                ],
            },
            {
                "model": model_name,
                "representation": representation,
                "feature_count": feature_sets[representation].shape[1],
                **final_metrics,
            },
        ]
    )
    comparison.to_csv(
        output / "improved_metrics.csv",
        index=False,
        float_format="%.6f",
    )

    save_confusion_matrix(
        true_labels,
        predicted_labels,
        f"{model_name} ({representation} features)",
        output / "improved_confusion_matrix.png",
    )
    save_largest_confusions(
        true_labels,
        predicted_labels,
        output / "improved_largest_confusions.csv",
    )
    save_per_class_metrics(
        true_labels,
        predicted_labels,
        output / "per_class_metrics.csv",
    )

    selected_parameter_names = build_searches()[model_name][1]
    selection_summary = {
        "model": model_name,
        "representation": representation,
        "feature_count": int(feature_sets[representation].shape[1]),
        "best_parameters": {
            name: estimator.get_params()[name] for name in selected_parameter_names
        },
        "cross_validation_macro_f1": float(tuning_results.iloc[0]["mean_test_score"]),
        "test_fold": TEST_FOLD,
        "test_metrics": final_metrics,
    }
    (output / "model_selection.json").write_text(
        json.dumps(selection_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSelected model comparison on untouched ESC-50 fold 5:")
    print(comparison.to_string(index=False))
    print(f"\nSelected representation: {representation}")
    print(f"Selected estimator: {model_name}")
    return comparison


def parse_arguments() -> argparse.Namespace:
    """Parse dataset, result, and cache paths for the improved experiment."""

    parser = argparse.ArgumentParser(
        description="Tune improved ESC-50 classifiers without using test fold 5."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/ESC-50"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results"))
    parser.add_argument(
        "--baseline-cache",
        type=Path,
        default=Path("artifacts/cache/esc50_features.npz"),
    )
    parser.add_argument(
        "--improved-cache",
        type=Path,
        default=Path("artifacts/cache/esc50_improved_features.npz"),
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore compatible caches and extract both representations again.",
    )
    return parser.parse_args()


def main() -> None:
    """Run improved model selection from the command line."""

    arguments = parse_arguments()
    run_experiment(
        arguments.dataset,
        arguments.output,
        arguments.baseline_cache,
        arguments.improved_cache,
        force_features=arguments.force_features,
    )


if __name__ == "__main__":
    main()
