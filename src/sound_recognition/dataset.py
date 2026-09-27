"""ESC-50 metadata validation and dataset-wide feature extraction."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sound_recognition.audio import DEFAULT_AUDIO_CONFIG, AudioConfig, load_audio
from sound_recognition.features import extract_features, feature_names

REQUIRED_COLUMNS = {"filename", "fold", "target", "category", "esc10", "src_file"}


def load_esc50_metadata(dataset_directory: str | Path) -> pd.DataFrame:
    """Load ESC-50 metadata and add a validated absolute audio path per row."""

    dataset_path = Path(dataset_directory).expanduser().resolve()
    metadata_path = dataset_path / "meta" / "esc50.csv"
    audio_directory = dataset_path / "audio"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"ESC-50 metadata was not found: {metadata_path}")
    if not audio_directory.is_dir():
        raise FileNotFoundError(
            f"ESC-50 audio directory was not found: {audio_directory}"
        )

    metadata = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ESC-50 metadata is missing required columns: {missing}")

    metadata = metadata.copy()
    metadata["audio_path"] = metadata["filename"].map(
        lambda filename: audio_directory / filename
    )
    missing_files = [path for path in metadata["audio_path"] if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"ESC-50 is incomplete; the first missing file is: {missing_files[0]}"
        )

    return metadata


def extract_dataset_features(
    metadata: pd.DataFrame,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float32]:
    """Extract one feature row for each metadata row in its original order."""

    if "audio_path" not in metadata:
        raise ValueError("Metadata must contain the validated 'audio_path' column.")

    total = len(metadata)
    number_of_features = len(feature_names(config))
    matrix = np.empty((total, number_of_features), dtype=np.float32)

    for position, audio_path in enumerate(metadata["audio_path"]):
        # Keeping loading and extraction together guarantees that notebooks,
        # experiments, and the future API use the same preparation operations.
        signal = load_audio(audio_path, config)
        matrix[position] = extract_features(signal, config)
        if progress_callback is not None:
            progress_callback(position + 1, total)

    return matrix
