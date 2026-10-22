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
const serviceStatus = document.querySelector("#service-status");
const serviceStatusText = document.querySelector("#service-status-text");

let previewUrl = null;

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

async function checkService() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error("Проверка состояния модели завершилась ошибкой.");
    }
    const payload = await response.json();
    serviceStatus.classList.add("is-online");
    serviceStatusText.textContent = `Модель готова · классов: ${payload.classes_count}`;
  } catch {
    serviceStatus.classList.add("is-offline");
    serviceStatusText.textContent = "Модель недоступна";
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

window.addEventListener("beforeunload", releasePreviewUrl);
checkService();
