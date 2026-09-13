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

## Dataset Exploration

Start JupyterLab:

```bash
uv run jupyter lab
```

Open `notebooks/01_esc50_exploration.ipynb`. The notebook checks the metadata and audio properties, provides playable examples, and displays a waveform and spectrogram.

## Project Structure

```text
sound-recognition/
├── data/                 # Local datasets excluded from Git
├── notebooks/            # Data exploration and experiments
├── src/sound_recognition # Reusable application code
├── pyproject.toml        # Project metadata and dependencies
└── uv.lock               # Locked dependency versions
```
