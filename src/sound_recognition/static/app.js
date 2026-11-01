"use strict";

const ALLOWED_EXTENSIONS = new Set(["wav", "flac", "ogg"]);
const LABEL_TRANSLATIONS = new Map([
  ["airplane", "Самолёт"],
  ["breathing", "Дыхание"],
  ["brushing_teeth", "Чистка зубов"],
  ["can_opening", "Открывание банки"],
  ["car_horn", "Автомобильный гудок"],
  ["cat", "Кошка"],
  ["chainsaw", "Бензопила"],
  ["chirping_birds", "Пение птиц"],
  ["church_bells", "Церковные колокола"],
  ["clapping", "Аплодисменты"],
  ["clock_alarm", "Будильник"],
  ["clock_tick", "Тиканье часов"],
  ["coughing", "Кашель"],
  ["cow", "Корова"],
  ["crackling_fire", "Треск огня"],
  ["crickets", "Сверчки"],
  ["crow", "Ворона"],
  ["crying_baby", "Плач ребёнка"],
  ["dog", "Собака"],
  ["door_wood_creaks", "Скрип деревянной двери"],
  ["door_wood_knock", "Стук в деревянную дверь"],
  ["drinking_sipping", "Питьё"],
  ["engine", "Двигатель"],
  ["fireworks", "Фейерверк"],
  ["footsteps", "Шаги"],
  ["frog", "Лягушка"],
  ["glass_breaking", "Разбитое стекло"],
  ["hand_saw", "Ручная пила"],
  ["helicopter", "Вертолёт"],
  ["hen", "Курица"],
  ["insects", "Насекомые"],
  ["keyboard_typing", "Печать на клавиатуре"],
  ["laughing", "Смех"],
  ["mouse_click", "Щелчок мыши"],
  ["pig", "Свинья"],
  ["pouring_water", "Льющаяся вода"],
  ["rain", "Дождь"],
  ["rooster", "Петух"],
  ["sea_waves", "Морские волны"],
  ["sheep", "Овца"],
  ["siren", "Сирена"],
  ["sneezing", "Чихание"],
  ["snoring", "Храп"],
  ["thunderstorm", "Гроза"],
  ["toilet_flush", "Смыв унитаза"],
  ["train", "Поезд"],
  ["vacuum_cleaner", "Пылесос"],
  ["washing_machine", "Стиральная машина"],
  ["water_drops", "Капли воды"],
  ["wind", "Ветер"],
]);

// The model uses only the first five seconds of a recording. Visualizing the
// same interval keeps the charts consistent with the audio passed to inference.
const VISUALIZATION_DURATION_SECONDS = 5;
const VISUALIZATION_SAMPLE_RATE = 22_050;
const FFT_SIZE = 1024;
const SPECTROGRAM_FRAME_COUNT = 128;
const SPECTROGRAM_BAND_COUNT = 96;
const SPECTROGRAM_DYNAMIC_RANGE_DB = 80;

const form = document.querySelector("#prediction-form");
const maxUploadBytes = Number(document.body.dataset.maxUploadBytes);
const fileInput = document.querySelector("#audio-file");
const dropZone = document.querySelector("#drop-zone");
const filePreview = document.querySelector("#file-preview");
const fileName = document.querySelector("#file-name");
const fileSize = document.querySelector("#file-size");
const removeFileButton = document.querySelector("#remove-file");
const audioPreview = document.querySelector("#audio-preview");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = submitButton.querySelector(".button-label");
const message = document.querySelector("#message");
const resultsEmpty = document.querySelector("#results-empty");
const resultsContent = document.querySelector("#results-content");
const primaryLabel = document.querySelector("#primary-label");
const primaryConfidence = document.querySelector("#primary-confidence");
const predictionList = document.querySelector("#prediction-list");
const processingTime = document.querySelector("#processing-time");
const predictionTemplate = document.querySelector("#prediction-item-template");
const visualizationPanel = document.querySelector("#visualization-panel");
const visualizationStatus = document.querySelector("#visualization-status");
const chartGrid = document.querySelector("#chart-grid");
const waveformCanvas = document.querySelector("#waveform-canvas");
const spectrogramCanvas = document.querySelector("#spectrogram-canvas");
const supportedClassList = document.querySelector("#supported-class-list");

let previewUrl = null;
let visualizationGeneration = 0;
let visualizationData = null;

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${formatDecimal(bytes / 1024)} КБ`;
  }
  return `${formatDecimal(bytes / (1024 * 1024))} МБ`;
}

function formatLabel(label) {
  // Translation stays in the presentation layer so the API and model artifact
  // continue to use the original stable ESC-50 class identifiers.
  return LABEL_TRANSLATIONS.get(label) ?? label.replaceAll("_", " ");
}

function formatDecimal(value) {
  return value.toLocaleString("ru-RU", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function formatChartDecimal(value) {
  // Axis labels use English notation independently of the Russian interface.
  // A fixed decimal place keeps corresponding ticks aligned between charts.
  return value.toLocaleString("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function confidenceText(confidence) {
  return `Уверенность: ${formatDecimal(confidence * 100)}%`;
}

function showMessage(text) {
  message.textContent = text;
  message.hidden = false;
}

function clearMessage() {
  message.textContent = "";
  message.hidden = true;
}

function clearResults() {
  resultsContent.hidden = true;
  resultsEmpty.hidden = false;
  primaryLabel.textContent = "";
  primaryConfidence.textContent = "";
  predictionList.replaceChildren();
  processingTime.textContent = "";
}

function prepareCanvas(canvas) {
  // Canvas bitmap dimensions are multiplied by the device pixel ratio while
  // drawing still uses CSS pixels. This keeps axes and curves sharp on HiDPI
  // displays without changing their responsive layout size.
  const width = Math.max(280, Math.floor(canvas.clientWidth));
  const height = Math.max(180, Math.floor(canvas.clientHeight));
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(width * pixelRatio);
  canvas.height = Math.floor(height * pixelRatio);

  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function drawChartAxes(
  context,
  plot,
  durationSeconds,
  { topLabel, middleLabel, bottomLabel, verticalTitle },
) {
  context.strokeStyle = "#dce3e4";
  context.lineWidth = 1;
  context.fillStyle = "#66727d";
  context.font = "11px Inter, system-ui, sans-serif";

  [0, 0.5, 1].forEach((position) => {
    const y = plot.top + plot.height * position;
    context.beginPath();
    context.moveTo(plot.left, y + 0.5);
    context.lineTo(plot.left + plot.width, y + 0.5);
    context.stroke();
  });

  context.textAlign = "right";
  context.fillText(topLabel, plot.left - 8, plot.top + 4);
  context.fillText(middleLabel, plot.left - 8, plot.top + plot.height / 2 + 4);
  context.fillText(bottomLabel, plot.left - 8, plot.top + plot.height + 4);

  // The vertical title names the physical quantity, while the three tick labels
  // show its numeric range. Keeping both in English follows common plot notation.
  context.save();
  context.translate(12, plot.top + plot.height / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText(verticalTitle, 0, 0);
  context.restore();

  context.textAlign = "left";
  context.fillText("0 s", plot.left, plot.top + plot.height + 20);
  context.textAlign = "right";
  context.fillText(
    `${formatChartDecimal(durationSeconds)} s`,
    plot.left + plot.width,
    plot.top + plot.height + 20,
  );
  context.textAlign = "center";
  context.fillText("Time", plot.left + plot.width / 2, plot.top + plot.height + 36);
}

function drawWaveform(signal, sampleRate) {
  const { context, width, height } = prepareCanvas(waveformCanvas);
  const plot = { left: 64, top: 14, width: width - 76, height: height - 58 };
  const durationSeconds = signal.length / sampleRate;
  drawChartAxes(context, plot, durationSeconds, {
    topLabel: "+1.0",
    middleLabel: "0.0",
    bottomLabel: "−1.0",
    verticalTitle: "Amplitude",
  });

  // Each horizontal pixel represents a range of samples. Drawing its minimum
  // and maximum preserves short peaks that a single-sample preview would miss.
  context.strokeStyle = "#0b6c6d";
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 0; x < plot.width; x += 1) {
    const start = Math.floor((x / plot.width) * signal.length);
    const end = Math.max(start + 1, Math.floor(((x + 1) / plot.width) * signal.length));
    let minimum = 1;
    let maximum = -1;
    for (let index = start; index < Math.min(end, signal.length); index += 1) {
      minimum = Math.min(minimum, signal[index]);
      maximum = Math.max(maximum, signal[index]);
    }
    const top = plot.top + ((1 - maximum) / 2) * plot.height;
    const bottom = plot.top + ((1 - minimum) / 2) * plot.height;
    const canvasX = plot.left + x + 0.5;
    context.moveTo(canvasX, top);
    context.lineTo(canvasX, bottom);
  }
  context.stroke();
}

function calculatePowerSpectrum(signal, offset) {
  const real = new Float64Array(FFT_SIZE);
  const imaginary = new Float64Array(FFT_SIZE);

  for (let index = 0; index < FFT_SIZE; index += 1) {
    const sample = signal[offset + index] ?? 0;
    // The Hann window makes the frame edges meet smoothly and reduces spectral
    // leakage, so narrow frequency components remain easier to distinguish.
    const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (FFT_SIZE - 1));
    real[index] = sample * window;
  }

  // Reorder input into bit-reversed positions before the iterative radix-2 FFT.
  // FFT_SIZE is fixed to a power of two, which keeps this browser-only transform
  // fast enough to run immediately after a user selects a recording.
  for (let index = 1, reversed = 0; index < FFT_SIZE; index += 1) {
    let bit = FFT_SIZE >> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
    }
  }

  for (let size = 2; size <= FFT_SIZE; size *= 2) {
    const angle = (-2 * Math.PI) / size;
    const stepReal = Math.cos(angle);
    const stepImaginary = Math.sin(angle);
    for (let start = 0; start < FFT_SIZE; start += size) {
      let twiddleReal = 1;
      let twiddleImaginary = 0;
      for (let index = 0; index < size / 2; index += 1) {
        const even = start + index;
        const odd = even + size / 2;
        const oddReal = real[odd] * twiddleReal - imaginary[odd] * twiddleImaginary;
        const oddImaginary =
          real[odd] * twiddleImaginary + imaginary[odd] * twiddleReal;
        real[odd] = real[even] - oddReal;
        imaginary[odd] = imaginary[even] - oddImaginary;
        real[even] += oddReal;
        imaginary[even] += oddImaginary;

        const nextReal = twiddleReal * stepReal - twiddleImaginary * stepImaginary;
        twiddleImaginary = twiddleReal * stepImaginary + twiddleImaginary * stepReal;
        twiddleReal = nextReal;
      }
    }
  }

  const power = new Float64Array(FFT_SIZE / 2);
  for (let index = 0; index < power.length; index += 1) {
    power[index] = real[index] ** 2 + imaginary[index] ** 2;
  }
  return power;
}

function calculateSpectrogram(signal, sampleRate) {
  const values = new Float32Array(SPECTROGRAM_FRAME_COUNT * SPECTROGRAM_BAND_COUNT);
  const availableTravel = Math.max(0, signal.length - FFT_SIZE);
  const hop = availableTravel / Math.max(1, SPECTROGRAM_FRAME_COUNT - 1);
  let maximumPower = 0;

  for (let frame = 0; frame < SPECTROGRAM_FRAME_COUNT; frame += 1) {
    const offset = Math.floor(frame * hop);
    const power = calculatePowerSpectrum(signal, offset);
    for (let band = 0; band < SPECTROGRAM_BAND_COUNT; band += 1) {
      const firstBin = Math.floor((band / SPECTROGRAM_BAND_COUNT) * power.length);
      const lastBin = Math.max(
        firstBin + 1,
        Math.floor(((band + 1) / SPECTROGRAM_BAND_COUNT) * power.length),
      );
      let total = 0;
      for (let bin = firstBin; bin < lastBin; bin += 1) {
        total += power[bin];
      }
      const average = total / (lastBin - firstBin);
      values[frame * SPECTROGRAM_BAND_COUNT + band] = average;
      maximumPower = Math.max(maximumPower, average);
    }
  }

  // Convert power to a relative decibel scale. Values below the selected
  // dynamic range are clipped so quiet numerical noise does not dominate color.
  for (let index = 0; index < values.length; index += 1) {
    const decibels = 10 * Math.log10((values[index] + Number.EPSILON) / (maximumPower + Number.EPSILON));
    values[index] = Math.max(0, 1 + decibels / SPECTROGRAM_DYNAMIC_RANGE_DB);
  }

  return {
    values,
    frameCount: SPECTROGRAM_FRAME_COUNT,
    bandCount: SPECTROGRAM_BAND_COUNT,
    maximumFrequency: sampleRate / 2,
  };
}

function spectrogramColor(value) {
  // This five-stop palette moves from dark blue through teal and yellow to the
  // interface accent. Linear interpolation avoids one DOM or canvas gradient
  // allocation for every time-frequency cell.
  const stops = [
    [17, 33, 58],
    [11, 108, 109],
    [74, 163, 137],
    [240, 194, 91],
    [230, 95, 60],
  ];
  const position = Math.min(1, Math.max(0, value)) * (stops.length - 1);
  const lowerIndex = Math.min(stops.length - 2, Math.floor(position));
  const fraction = position - lowerIndex;
  const lower = stops[lowerIndex];
  const upper = stops[lowerIndex + 1];
  const channel = (index) => Math.round(lower[index] + (upper[index] - lower[index]) * fraction);
  return `rgb(${channel(0)} ${channel(1)} ${channel(2)})`;
}

function drawSpectrogram(spectrogram, durationSeconds) {
  const { context, width, height } = prepareCanvas(spectrogramCanvas);
  const plot = { left: 68, top: 14, width: width - 80, height: height - 58 };
  const maximumKilohertz = spectrogram.maximumFrequency / 1000;
  drawChartAxes(context, plot, durationSeconds, {
    topLabel: `${formatChartDecimal(maximumKilohertz)} kHz`,
    middleLabel: `${formatChartDecimal(maximumKilohertz / 2)} kHz`,
    bottomLabel: "0 Hz",
    verticalTitle: "Frequency",
  });

  const cellWidth = plot.width / spectrogram.frameCount;
  const cellHeight = plot.height / spectrogram.bandCount;
  for (let frame = 0; frame < spectrogram.frameCount; frame += 1) {
    for (let band = 0; band < spectrogram.bandCount; band += 1) {
      const value = spectrogram.values[frame * spectrogram.bandCount + band];
      context.fillStyle = spectrogramColor(value);
      // Low frequencies are stored first but are drawn at the bottom of the
      // chart, matching the conventional orientation of a spectrogram.
      const y = plot.top + plot.height - (band + 1) * cellHeight;
      context.fillRect(plot.left + frame * cellWidth, y, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

function redrawVisualizations() {
  if (visualizationData === null || visualizationPanel.hidden) {
    return;
  }
  drawWaveform(visualizationData.signal, visualizationData.sampleRate);
  drawSpectrogram(
    visualizationData.spectrogram,
    visualizationData.signal.length / visualizationData.sampleRate,
  );
}

function clearVisualizations() {
  visualizationGeneration += 1;
  visualizationData = null;
  visualizationPanel.hidden = true;
  chartGrid.hidden = false;
}

async function renderAudioVisualizations(file) {
  const generation = ++visualizationGeneration;
  visualizationPanel.hidden = false;
  chartGrid.hidden = true;
  visualizationStatus.textContent = "Декодируем запись и строим графики…";

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (AudioContextClass === undefined) {
    visualizationStatus.textContent = "Этот браузер не поддерживает визуализацию аудио.";
    return;
  }

  // Request the same sample rate as the server pipeline. Besides making the
  // displayed frequency scale reproducible across browsers, this ensures the
  // spectrogram represents the signal bandwidth that is available to the model.
  const audioContext = new AudioContextClass({ sampleRate: VISUALIZATION_SAMPLE_RATE });
  try {
    const encodedAudio = await file.arrayBuffer();
    const decodedAudio = await audioContext.decodeAudioData(encodedAudio.slice(0));
    if (generation !== visualizationGeneration) {
      return;
    }

    const sampleCount = Math.min(
      decodedAudio.length,
      Math.round(decodedAudio.sampleRate * VISUALIZATION_DURATION_SECONDS),
    );
    const signal = new Float32Array(sampleCount);
    for (let channel = 0; channel < decodedAudio.numberOfChannels; channel += 1) {
      const channelData = decodedAudio.getChannelData(channel);
      for (let index = 0; index < sampleCount; index += 1) {
        signal[index] += channelData[index] / decodedAudio.numberOfChannels;
      }
    }

    // Match the server preprocessing: non-silent signals are peak-normalized.
    // The charts therefore show structure clearly even for quiet recordings.
    let peak = 0;
    for (const sample of signal) {
      peak = Math.max(peak, Math.abs(sample));
    }
    if (peak > 0) {
      const scale = 0.95 / peak;
      for (let index = 0; index < signal.length; index += 1) {
        signal[index] *= scale;
      }
    }

    visualizationData = {
      signal,
      sampleRate: decodedAudio.sampleRate,
      spectrogram: calculateSpectrogram(signal, decodedAudio.sampleRate),
    };
    chartGrid.hidden = false;
    visualizationStatus.textContent = `Анализируемый фрагмент: первые ${formatDecimal(
      signal.length / decodedAudio.sampleRate,
    )} с записи. Если файл содержит несколько каналов, они объединяются в один сигнал.`;
    requestAnimationFrame(redrawVisualizations);
  } catch {
    // Browser codec support can differ from the server decoder. Prediction is
    // intentionally left available even if only the local charts cannot be built.
    chartGrid.hidden = true;
    visualizationStatus.textContent =
      "Браузер не смог построить графики, но запись всё ещё можно отправить модели.";
  } finally {
    await audioContext.close();
  }
}

function setBusy(isBusy) {
  fileInput.disabled = isBusy;
  removeFileButton.disabled = isBusy;
  submitButton.disabled = isBusy || fileInput.files.length === 0;
  submitButton.classList.toggle("is-loading", isBusy);
  buttonLabel.textContent = isBusy ? "Анализируем запись…" : "Распознать звук";
  form.setAttribute("aria-busy", String(isBusy));
}

function releasePreviewUrl() {
  // Object URLs hold the selected file in memory. Revoke the previous one when
  // the user chooses or removes a file so repeated selections do not leak memory.
  if (previewUrl !== null) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function resetSelection() {
  releasePreviewUrl();
  clearVisualizations();
  fileInput.value = "";
  audioPreview.removeAttribute("src");
  audioPreview.load();
  filePreview.hidden = true;
  dropZone.hidden = false;
  submitButton.disabled = true;
  clearMessage();
  clearResults();
}

function validationError(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ALLOWED_EXTENSIONS.has(extension)) {
    return "Выберите аудиофайл в формате WAV, FLAC или OGG.";
  }
  if (file.size === 0) {
    return "Выбранный файл пуст.";
  }
  if (file.size > maxUploadBytes) {
    return `Размер файла превышает ограничение ${formatBytes(maxUploadBytes)}.`;
  }
  return null;
}

function showSelectedFile(file) {
  const error = validationError(file);
  if (error !== null) {
    resetSelection();
    showMessage(error);
    return;
  }

  releasePreviewUrl();
  previewUrl = URL.createObjectURL(file);
  audioPreview.src = previewUrl;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  dropZone.hidden = true;
  filePreview.hidden = false;
  submitButton.disabled = false;
  clearMessage();
  clearResults();
  renderAudioVisualizations(file);
}

function renderResults(payload) {
  const [firstPrediction] = payload.predictions;
  if (firstPrediction === undefined) {
    throw new Error("Сервер не вернул ни одного результата.");
  }

  primaryLabel.textContent = formatLabel(firstPrediction.label);
  primaryConfidence.textContent = confidenceText(firstPrediction.confidence);
  predictionList.replaceChildren();

  payload.predictions.forEach((prediction, index) => {
    const item = predictionTemplate.content.cloneNode(true);
    item.querySelector(".prediction-item__rank").textContent = String(index + 1).padStart(2, "0");
    item.querySelector(".prediction-item__label").textContent = formatLabel(prediction.label);
    item.querySelector(".prediction-item__confidence").textContent = `${formatDecimal(
      prediction.confidence * 100,
    )}%`;
    const progress = item.querySelector(".prediction-item__track");
    const percentage = Math.min(100, Math.max(0, prediction.confidence * 100));
    progress.setAttribute("aria-valuenow", percentage.toFixed(1));
    progress.setAttribute(
      "aria-label",
      `Уверенность для класса ${formatLabel(prediction.label)}: ${formatDecimal(percentage)}%`,
    );
    item.querySelector(".prediction-item__bar").style.width = `${percentage}%`;
    predictionList.append(item);
  });

  processingTime.textContent = `Время обработки моделью: ${formatDecimal(
    payload.processing_ms,
  )} мс`;
  resultsEmpty.hidden = true;
  resultsContent.hidden = false;
}

function errorDetail(status, payload, fallback) {
  // Known API errors are translated without exposing implementation details.
  if (status === 413) {
    return `Файл слишком большой. Максимальный размер — ${formatBytes(maxUploadBytes)}.`;
  }
  if (status === 415) {
    return "Неподдерживаемый формат. Выберите файл WAV, FLAC или OGG.";
  }
  if (status === 422) {
    return "Не удалось обработать аудиофайл. Проверьте, что он не пуст и не повреждён.";
  }
  if (Array.isArray(payload?.detail)) {
    return "Сервер отклонил некорректный запрос.";
  }
  return fallback;
}

async function loadSupportedClasses() {
  try {
    const response = await fetch("/classes", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("Не удалось загрузить список классов.");
    }
    const payload = await response.json();
    const classItems = payload.classes.map((label) => {
      const item = document.createElement("span");
      item.className = "class-catalogue__item";
      item.textContent = formatLabel(label);
      return item;
    });
    supportedClassList.replaceChildren(...classItems);
  } catch {
    supportedClassList.textContent = "Список классов доступен после запуска модели.";
  }
}

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file !== undefined) {
    showSelectedFile(file);
  }
});

removeFileButton.addEventListener("click", resetSelection);

// Drag events are handled explicitly because the hidden native file input only
// receives files selected through its browser dialog.
["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file === undefined) {
    return;
  }

  // DataTransfer creates a real FileList so the dropped file is submitted by
  // the same FormData code path as a file chosen through the native dialog.
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = fileInput.files;
  if (file === undefined) {
    showMessage("Сначала выберите аудиозапись.");
    return;
  }

  clearMessage();
  clearResults();
  setBusy(true);

  try {
    const body = new FormData();
    body.append("file", file, file.name);
    const response = await fetch("/predict", { method: "POST", body });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      // A non-JSON response indicates an unexpected server or proxy failure.
    }

    if (!response.ok) {
      throw new Error(
        errorDetail(
          response.status,
          payload,
          `Не удалось распознать запись: HTTP ${response.status}.`,
        ),
      );
    }
    renderResults(payload);
  } catch (error) {
    const text =
      error instanceof TypeError
        ? "Не удалось связаться с сервером. Проверьте, что приложение запущено."
        : error instanceof Error
          ? error.message
          : "Не удалось обработать запись.";
    showMessage(text);
  } finally {
    setBusy(false);
  }
});

// Redraw from cached numeric data when responsive CSS changes a chart width.
// requestAnimationFrame coalesces multiple ResizeObserver notifications into a
// single paint and avoids repeating the FFT calculation during window resizing.
let pendingRedraw = null;
const visualizationResizeObserver = new ResizeObserver(() => {
  if (pendingRedraw !== null) {
    cancelAnimationFrame(pendingRedraw);
  }
  pendingRedraw = requestAnimationFrame(() => {
    pendingRedraw = null;
    redrawVisualizations();
  });
});
visualizationResizeObserver.observe(waveformCanvas);
visualizationResizeObserver.observe(spectrogramCanvas);

window.addEventListener("beforeunload", releasePreviewUrl);
loadSupportedClasses();
