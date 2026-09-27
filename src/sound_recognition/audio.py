"""Audio loading and preprocessing shared by training and inference."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray


@dataclass(frozen=True)
class AudioConfig:
    """Parameters that define the single audio format used by the project."""

    sample_rate: int = 22_050
    duration_seconds: float = 5.0
    peak_amplitude: float = 0.95
    n_fft: int = 2_048
    hop_length: int = 512
    n_mfcc: int = 20

    @property
    def number_of_samples(self) -> int:
        """Return the exact signal length expected by the feature extractor."""

        return round(self.sample_rate * self.duration_seconds)


DEFAULT_AUDIO_CONFIG = AudioConfig()


def prepare_audio(
    signal: NDArray[np.floating],
    source_sample_rate: int,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> NDArray[np.float32]:
    """Convert an in-memory signal to mono, resample, resize, and normalize it.

    Two-dimensional input is expected in the sample-major layout returned by
    ``soundfile``: ``(number_of_samples, number_of_channels)``. Short signals are
    padded with silence at the end, while long signals are cropped. Peak
    normalization preserves the shape of the waveform and leaves silence intact.
    """

    if source_sample_rate <= 0:
        raise ValueError("The source sample rate must be a positive integer.")

    audio = np.asarray(signal, dtype=np.float32)
    if audio.ndim == 2:
        # Averaging all channels provides a deterministic mono downmix without
        # selecting one channel and silently discarding information from others.
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError("Audio must be a one-dimensional or two-dimensional array.")

    if audio.size == 0:
        raise ValueError("Audio must contain at least one sample.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite sample values.")

    if source_sample_rate != config.sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=source_sample_rate,
            target_sr=config.sample_rate,
        )

    # ``fix_length`` applies zero-padding or right-side cropping, so every file
    # produces the same number of time frames and therefore the same feature size.
    audio = librosa.util.fix_length(audio, size=config.number_of_samples)

    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio * (config.peak_amplitude / peak)

    return np.asarray(audio, dtype=np.float32)


def load_audio(
    path: str | Path,
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> NDArray[np.float32]:
    """Read an audio file and return it in the project's standard format."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    signal, source_sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
    return prepare_audio(signal, source_sample_rate, config)
