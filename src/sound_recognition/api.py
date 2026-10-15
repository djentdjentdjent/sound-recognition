"""HTTP API for environmental sound classification."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Protocol

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from sound_recognition.audio import SUPPORTED_AUDIO_EXTENSIONS, AudioLoadingError
from sound_recognition.model_training import DEFAULT_MODEL_PATH
from sound_recognition.prediction import AudioClassifier, PredictionResult

DEFAULT_TOP_K = 3
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024

# Browsers and command-line clients do not always agree on one media type for
# the same audio container. The API accepts the established aliases below, but
# still checks the filename extension and decodes the container before inference.
CONTENT_TYPES_BY_EXTENSION = {
    ".flac": frozenset({"audio/flac", "audio/x-flac"}),
    ".ogg": frozenset({"application/ogg", "audio/ogg"}),
    ".wav": frozenset({"audio/vnd.wave", "audio/wav", "audio/wave", "audio/x-wav"}),
}
GENERIC_BINARY_CONTENT_TYPE = "application/octet-stream"


class Predictor(Protocol):
    """Describe the classifier interface used by HTTP route handlers."""

    classes: tuple[str, ...]

    def predict(self, audio_path: str | Path, top_k: int = 3) -> PredictionResult:
        """Classify one audio file and return its highest-ranked classes."""


class HealthResponse(BaseModel):
    """Report that the server and its model are ready to accept requests."""

    status: Literal["ok"]
    model_loaded: bool
    classes_count: int = Field(ge=1)


class ClassesResponse(BaseModel):
    """List every class label supported by the loaded model."""

    count: int = Field(ge=1)
    classes: list[str]


class ClassPredictionResponse(BaseModel):
    """Represent one item in a prediction ranking."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Describe the complete response returned for an uploaded recording."""

    filename: str
    predictions: list[ClassPredictionResponse]
    processing_ms: float = Field(gt=0.0)


class ErrorResponse(BaseModel):
    """Document project errors and FastAPI's missing-field validation errors."""

    detail: str | list[dict[str, object]]


def _validate_upload_metadata(upload: UploadFile) -> str:
    """Validate the filename and declared media type before reading the body."""

    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension. Supported extensions: {supported}.",
        )

    # Parameters such as ``codecs=vorbis`` may follow a semicolon. They do not
    # change the container type and therefore are removed before comparison.
    content_type = (upload.content_type or "").split(";", maxsplit=1)[0].lower()
    allowed_content_types = CONTENT_TYPES_BY_EXTENSION[extension]
    if content_type not in allowed_content_types | {GENERIC_BINARY_CONTENT_TYPE}:
        expected = ", ".join(sorted(allowed_content_types))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Media type '{content_type or '<missing>'}' does not match "
                f"the {extension} container. Expected one of: {expected}."
            ),
        )

    return extension


async def _copy_upload_with_limit(
    upload: UploadFile,
    destination: Path,
    max_upload_bytes: int,
) -> None:
    """Stream an upload to disk and stop as soon as its size limit is exceeded."""

    total_bytes = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=(
                        "Uploaded file is too large. "
                        f"Maximum size is {max_upload_bytes} bytes."
                    ),
                )
            output.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file is empty.",
        )


def create_app(
    *,
    classifier: Predictor | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> FastAPI:
    """Create an application whose classifier is loaded once during startup.

    Passing a classifier is useful for isolated API tests. In normal operation,
    the startup lifespan loads the versioned project artifact and stores the
    reusable predictor in application state for every subsequent request.
    """

    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive.")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Loading a serialized model is deliberately kept outside route handlers.
        # This makes startup fail early for an invalid artifact and avoids paying
        # the deserialization cost for every uploaded recording.
        application.state.classifier = (
            classifier if classifier is not None else AudioClassifier(model_path)
        )
        application.state.max_upload_bytes = max_upload_bytes
        yield

    application = FastAPI(
        title="Environmental Sound Recognition API",
        summary="Classify short WAV, FLAC, and OGG environmental recordings.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Check whether the API and classifier are ready",
    )
    def health(request: Request) -> HealthResponse:
        loaded_classifier: Predictor = request.app.state.classifier
        return HealthResponse(
            status="ok",
            model_loaded=True,
            classes_count=len(loaded_classifier.classes),
        )

    @application.get(
        "/classes",
        response_model=ClassesResponse,
        summary="List the sound classes supported by the model",
    )
    def classes(request: Request) -> ClassesResponse:
        loaded_classifier: Predictor = request.app.state.classifier
        labels = list(loaded_classifier.classes)
        return ClassesResponse(count=len(labels), classes=labels)

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        },
        summary="Return the three highest-ranked sound classes",
    )
    async def predict(
        request: Request,
        file: Annotated[
            UploadFile,
            File(description="A WAV, FLAC, or OGG recording up to 10 MiB."),
        ],
    ) -> PredictionResponse:
        extension = _validate_upload_metadata(file)
        temporary_path: Path | None = None

        try:
            # The original extension is retained because the shared audio loader
            # uses it as an inexpensive first validation step before decoding.
            with tempfile.NamedTemporaryFile(
                prefix="sound-recognition-",
                suffix=extension,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            await _copy_upload_with_limit(
                file,
                temporary_path,
                request.app.state.max_upload_bytes,
            )
            loaded_classifier: Predictor = request.app.state.classifier

            # Feature extraction is CPU-bound synchronous work. Running it in
            # FastAPI's worker pool keeps the event loop responsive while a
            # recording is decoded and converted into model features.
            result = await run_in_threadpool(
                loaded_classifier.predict,
                temporary_path,
                DEFAULT_TOP_K,
            )
        except (AudioLoadingError, ValueError) as error:
            # The shared pipeline uses ValueError for decoded signals that are
            # empty or contain invalid numeric samples. Such input is a client
            # error just like a container that SoundFile cannot decode.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return PredictionResponse(
            filename=file.filename or "",
            predictions=[
                ClassPredictionResponse(
                    label=prediction.label,
                    confidence=prediction.confidence,
                )
                for prediction in result.predictions
            ],
            processing_ms=result.processing_ms,
        )

    return application


app = create_app()


def main() -> None:
    """Run the development server with automatic interactive documentation."""

    uvicorn.run(
        "sound_recognition.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
