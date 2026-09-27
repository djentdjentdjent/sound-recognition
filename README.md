# Environmental Sound Recognition

A web application for automatic environmental sound classification using machine learning.

## Goal

The application will allow a user to upload a short audio recording and receive the three most probable sound classes with confidence scores.

## Planned MVP

- Upload a short audio file.
- Play the uploaded recording.
- Classify the dominant environmental sound.
- Display the three most probable classes.
- Display confidence scores.
- Show the waveform and spectrogram.

## Dataset

The project uses the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). It contains 2,000 five-second environmental audio recordings divided into 50 balanced classes and five predefined folds.

| Property | Value |
|---|---:|
| Recordings | 2,000 |
| Classes | 50 |
| Recordings per class | 40 |
| Folds | 5 |
| Recordings per fold | 400 |
| Format | WAV, 16-bit PCM |
| Sample rate | 44.1 kHz |
| Channels | Mono |
| Duration | 5 seconds |

Dataset files are stored locally in `data/ESC-50` and are not included in the project repository.

## Environment Setup

The project requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for Python and dependency management.

Install the project environment:

```bash
uv python install 3.12
uv sync
```

The ESC-50 archive must be downloaded separately from the official dataset repository and extracted to:

```text
data/ESC-50/
```

The following paths must exist before running the exploration notebook:

```text
data/ESC-50/audio/
data/ESC-50/meta/esc50.csv
```

## Audio Processing Pipeline

Every recording is transformed by the same reusable pipeline:

1. Downmix all channels to mono.
2. Resample to 22,050 Hz.
3. Crop or zero-pad to exactly 5 seconds (110,250 samples).
4. Peak-normalize non-silent audio to an amplitude of 0.95.
5. Extract 154 statistical features: MFCC, first- and second-order MFCC
   deltas, chroma, spectral centroid, spectral bandwidth, spectral rolloff,
   zero-crossing rate, and RMS energy.

The implementation is in `src/sound_recognition/audio.py` and
`src/sound_recognition/features.py`. It is shared by dataset experiments and
will later be reused by the prediction service.

## Notebooks

Start JupyterLab:

```bash
uv run jupyter lab
```

- `notebooks/01_esc50_exploration.ipynb` checks metadata and audio properties,
  provides playable examples, and displays a waveform and spectrogram.
- `notebooks/02_audio_features_and_baselines.ipynb` checks extracted features on
  several classes, trains the first models, displays their metrics, and shows a
  normalized confusion matrix.

## Baseline Experiment

Run the complete experiment from the repository root:

```bash
uv run train-baselines
```

The first run extracts features from all 2,000 recordings and creates
`artifacts/cache/esc50_features.npz`. The cache is excluded from Git and makes
later runs substantially faster. Delete it or add `--force-features` to repeat
feature extraction from the WAV files.

The evaluation follows the predefined ESC-50 split: folds 1–4 contain 1,600
training recordings and fold 5 contains 400 test recordings. The current result
is:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| RBF support vector machine | 0.5525 | 0.5410 |
| Random forest | 0.5575 | 0.5301 |
| Logistic regression | 0.4950 | 0.4846 |
| Most frequent class | 0.0200 | 0.0008 |

Macro F1 is the primary comparison metric because it gives equal weight to every
sound class. By this metric, the RBF support vector machine is the most promising
of the initial models. Generated results are stored in `artifacts/results/`.

## Quality Checks

Run the automated tests and static checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests cover stereo-to-mono conversion, resampling, fixed-duration padding,
peak normalization, silent input, and fixed-length finite feature extraction.

## Project Structure

```text
sound-recognition/
├── artifacts/
│   ├── cache/            # Local feature cache excluded from Git
│   └── results/          # Metrics and confusion analysis
├── data/                 # Local datasets excluded from Git
├── notebooks/            # Data exploration and experiments
├── src/sound_recognition # Reusable audio and experiment code
├── tests/                # Automated pipeline tests
├── pyproject.toml        # Project metadata and tool settings
└── uv.lock               # Locked dependency versions
```
