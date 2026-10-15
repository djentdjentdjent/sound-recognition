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

The selected model uses an improved 424-value representation. It contains all
154 baseline values plus the mean and standard deviation of 128 log-mel bands
and seven spectral-contrast bands. The additional descriptors preserve more
frequency-distribution information than MFCC values alone.

The implementation is in `src/sound_recognition/audio.py` and
`src/sound_recognition/features.py`. It is shared by dataset experiments and
the prediction service.

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

## Improved Model Selection

Run the model-selection experiment from the repository root:

```bash
uv run train-improved
```

The command compares baseline and improved feature representations. It tunes an
RBF support vector machine and a random forest with four-fold cross-validation
on ESC-50 folds 1-4. Fold 5 is excluded from all parameter selection and used
exactly once to evaluate the selected configuration.

The selected configuration is an RBF support vector machine with `C=10` and
`gamma="scale"`, trained on the 424 improved features. Its results on fold 5 are:

| Model | Features | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Baseline RBF SVM | 154 | 0.5525 | 0.5410 |
| Selected RBF SVM | 424 | **0.6250** | **0.6037** |

The mean cross-validation macro F1 of the selected configuration is 0.5939.
Generated artifacts include the complete tuning table, final comparison,
per-class metrics, normalized confusion matrix, and largest confusion pairs.
The feature cache is excluded from Git and reused on subsequent runs.

## Application Model and Local Prediction

Train the selected configuration on ESC-50 folds 1-4 and recreate the versioned
application artifact:

```bash
uv run train-model
```

The command writes two files to `artifacts/model/`:

- `sound_classifier.joblib` contains the fitted scaler and RBF SVM together
  with the audio configuration, exact feature order, class order, and artifact
  schema versions required for inference.
- `model_metadata.json` is a human-readable description of the same model,
  including package versions, fold 5 metrics, and end-to-end timing results.

The saved model is loaded without retraining. Predict a supported WAV, FLAC, or
OGG recording with:

```bash
uv run predict-sound path/to/recording.wav
```

The JSON result contains the three highest-ranked ESC-50 classes, normalized
confidence values, and processing time. Confidence is calculated by applying
softmax to the SVM one-vs-rest decision scores, so the values are useful for
ranking but are not independently calibrated probabilities. Processing time
includes decoding, resampling, normalization, feature extraction, confidence
calculation, and ranking; model loading is excluded because the future server
will load the artifact once and reuse it.

The training command also validates the model on untouched fold 5. The saved
artifact retains accuracy `0.6250` and macro F1 `0.6037`. On the current
development machine, warm end-to-end inference averaged about 20 ms over 50
recordings. The first prediction in a new process can be slower because audio
libraries perform one-time initialization.

Missing files, unsupported extensions, damaged containers, incompatible model
versions, changed feature order, and invalid `top_k` values produce explicit
errors instead of partial predictions.

## HTTP API

Start the FastAPI development server from the repository root:

```bash
uv run serve-api
```

The server loads `artifacts/model/sound_classifier.joblib` once during startup
and reuses the same classifier for every request. Open the interactive Swagger
documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The API provides three endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the server and its model are ready. |
| `GET` | `/classes` | Return all 50 ESC-50 class labels. |
| `POST` | `/predict` | Upload one recording and receive the top three classes. |

Check server readiness:

```bash
curl http://127.0.0.1:8000/health
```

Classify a local recording:

```bash
curl -X POST \
  -F "file=@path/to/recording.wav;type=audio/wav" \
  http://127.0.0.1:8000/predict
```

The upload endpoint accepts WAV, FLAC, and OGG containers up to 10 MiB. It
checks the filename extension and declared media type, rejects empty or
oversized bodies, and then lets the shared audio loader validate the actual
container. HTTP 415 is returned for unsupported file types, HTTP 413 for files
above the size limit, and HTTP 422 for empty or damaged audio. A successful
response contains the original filename, three ranked labels with confidence
scores, and complete model processing time in milliseconds. Temporary upload
files are removed after both successful and failed requests.

## Quality Checks

Run the automated tests and static checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests cover stereo-to-mono conversion, resampling, fixed-duration padding,
peak normalization, silent input, fixed-length feature extraction, and finite
improved descriptors for both ordinary and silent audio. They also cover model
artifact loading, ranked confidence output, all three HTTP endpoints, upload
size and type limits, temporary-file cleanup, and rejection of unsupported or
damaged files. The repository currently contains 19 passing tests.

## Project Structure

```text
sound-recognition/
├── artifacts/
│   ├── cache/            # Local feature cache excluded from Git
│   ├── model/            # Versioned fitted model and readable metadata
│   └── results/          # Metrics and confusion analysis
├── data/                 # Local datasets excluded from Git
├── notebooks/            # Data exploration and experiments
├── src/sound_recognition # Reusable audio, model, experiment, and API code
├── tests/                # Automated pipeline tests
├── pyproject.toml        # Project metadata and tool settings
└── uv.lock               # Locked dependency versions
```
