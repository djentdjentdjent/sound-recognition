"""Extraction of compact statistical features from prepared audio."""

from collections.abc import Sequence

import librosa
import numpy as np
from numpy.typing import NDArray

from sound_recognition.audio import DEFAULT_AUDIO_CONFIG, AudioConfig

FEATURE_SCHEMA_VERSION = 1
IMPROVED_FEATURE_SCHEMA_VERSION = 1


def _summarize_frames(
    values: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Summarize every feature row by its mean and standard deviation."""

    means = np.mean(values, axis=1)
    standard_deviations = np.std(values, axis=1)
    return np.concatenate((means, standard_deviations)).astype(np.float32)


def extract_features(
    signal: NDArray[np.floating],
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> NDArray[np.float32]:
    """Convert one prepared waveform into a fixed-length feature vector.

    MFCC coefficients describe the spectral envelope, their first two deltas
    describe temporal changes, and the remaining descriptors capture pitch-class,
    energy, frequency-distribution, and zero-crossing characteristics. Frame-level
    values are reduced to means and standard deviations so classical estimators can
    process recordings as ordinary tabular rows.
    """

    audio = np.asarray(signal, dtype=np.float32)
    if audio.ndim != 1 or audio.size != config.number_of_samples:
        raise ValueError(
            "The signal must be a prepared one-dimensional array with "
            f"{config.number_of_samples} samples."
        )

    short_time_fourier_transform = librosa.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    magnitude = np.abs(short_time_fourier_transform)
    power = magnitude**2

    mel_spectrogram = librosa.feature.melspectrogram(
        S=power,
        sr=config.sample_rate,
        n_mels=128,
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=config.n_mfcc)

    # Delta features use neighbouring frames and expose changes that are hidden
    # when only the average spectral envelope is retained.
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)

    descriptors = (
        mfcc,
        mfcc_delta,
        mfcc_delta_delta,
        librosa.feature.chroma_stft(
            S=magnitude,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            # Fixed tuning avoids estimating pitch from recordings that contain
            # little or no tonal energy and makes extraction fully deterministic.
            tuning=0.0,
        ),
        librosa.feature.spectral_centroid(
            S=magnitude,
            sr=config.sample_rate,
        ),
        librosa.feature.spectral_bandwidth(
            S=magnitude,
            sr=config.sample_rate,
        ),
        librosa.feature.spectral_rolloff(
            S=magnitude,
            sr=config.sample_rate,
            roll_percent=0.85,
        ),
        librosa.feature.zero_crossing_rate(
            audio,
            frame_length=config.n_fft,
            hop_length=config.hop_length,
        ),
        librosa.feature.rms(
            S=magnitude,
            frame_length=config.n_fft,
            hop_length=config.hop_length,
        ),
    )

    vector = np.concatenate([_summarize_frames(item) for item in descriptors])
    if not np.isfinite(vector).all():
        raise ValueError("Feature extraction produced a non-finite value.")
    return vector


def extract_improved_features(
    signal: NDArray[np.floating],
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> NDArray[np.float32]:
    """Extract the baseline descriptors and a richer spectral representation.

    The baseline vector compresses the mel spectrum into MFCC coefficients. That
    representation is compact, but some information about energy in individual
    frequency bands is lost. The improved vector therefore appends mean and
    standard deviation values for every log-mel band. Spectral contrast is also
    included to describe the difference between peaks and valleys in octave-like
    frequency regions, which is useful for separating tonal and noisy sounds.
    """

    audio = np.asarray(signal, dtype=np.float32)
    if audio.ndim != 1 or audio.size != config.number_of_samples:
        raise ValueError(
            "The signal must be a prepared one-dimensional array with "
            f"{config.number_of_samples} samples."
        )

    baseline = extract_features(audio, config)
    short_time_fourier_transform = librosa.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    magnitude = np.abs(short_time_fourier_transform)
    power = magnitude**2

    mel_spectrogram = librosa.feature.melspectrogram(
        S=power,
        sr=config.sample_rate,
        n_mels=128,
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Spectral contrast requires at least one non-zero magnitude. Silent files
    # are represented by zeros so the pipeline remains finite and deterministic.
    if np.any(magnitude):
        # Six octave boundaries above ``fmin`` must stay below the Nyquist
        # frequency. Lower-rate inputs therefore need a smaller first boundary
        # than librosa's 200 Hz default, while the project rate keeps 200 Hz.
        contrast_fmin = min(200.0, config.sample_rate * 0.95 / (2**7))
        spectral_contrast = librosa.feature.spectral_contrast(
            S=magnitude,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            fmin=contrast_fmin,
        )
    else:
        spectral_contrast = np.zeros((7, magnitude.shape[1]), dtype=np.float32)

    vector = np.concatenate(
        (
            baseline,
            _summarize_frames(log_mel_spectrogram),
            _summarize_frames(spectral_contrast),
        )
    ).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Improved feature extraction produced a non-finite value.")
    return vector


def feature_names(config: AudioConfig = DEFAULT_AUDIO_CONFIG) -> Sequence[str]:
    """Return feature labels in exactly the same order as ``extract_features``."""

    groups = (
        ("mfcc", config.n_mfcc),
        ("mfcc_delta", config.n_mfcc),
        ("mfcc_delta2", config.n_mfcc),
        ("chroma", 12),
        ("spectral_centroid", 1),
        ("spectral_bandwidth", 1),
        ("spectral_rolloff", 1),
        ("zero_crossing_rate", 1),
        ("rms", 1),
    )
    names: list[str] = []
    for group_name, count in groups:
        for statistic in ("mean", "std"):
            names.extend(
                f"{group_name}_{index:02d}_{statistic}" for index in range(1, count + 1)
            )
    return names


def improved_feature_names(
    config: AudioConfig = DEFAULT_AUDIO_CONFIG,
) -> Sequence[str]:
    """Return labels for the improved vector in extraction order."""

    names = list(feature_names(config))
    for group_name, count in (("log_mel", 128), ("spectral_contrast", 7)):
        for statistic in ("mean", "std"):
            names.extend(
                f"{group_name}_{index:03d}_{statistic}" for index in range(1, count + 1)
            )
    return names
