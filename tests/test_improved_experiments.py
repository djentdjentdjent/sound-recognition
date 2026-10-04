"""Tests for leakage-free fold construction in model selection."""

import numpy as np
import pytest

from sound_recognition.improved_experiments import _training_fold_split


def test_training_fold_split_uses_each_training_fold_for_validation() -> None:
    """Folds 1-4 should produce four non-overlapping validation partitions."""

    folds = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    split = _training_fold_split(folds)

    partitions = list(split.split())

    assert len(partitions) == 4
    validation_indices = np.concatenate([validation for _, validation in partitions])
    assert sorted(validation_indices.tolist()) == list(range(len(folds)))


def test_training_fold_split_rejects_test_fold() -> None:
    """Fold 5 must never enter hyperparameter selection."""

    folds = np.asarray([1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="must not contain test fold 5"):
        _training_fold_split(folds)
