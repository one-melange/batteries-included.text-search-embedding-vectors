"""FastAPI surface for the Document Preparation page."""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from packages.vector_search.src.document_preparation.models import ComparisonRequest
from packages.vector_search.src.document_preparation.service import ComparisonService

service = ComparisonService()
app = FastAPI(title="Document Preparation API", version="0.1.0")

origins = os.environ.get(
    "DOCUMENT_PREPARATION_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/document-preparation/options")
def get_options():
    """Return the matrix metadata without loading model artifacts."""

    return service.options()


def _validation_message(exc: ValidationError) -> str:
    error = exc.errors(include_url=False)[0]
    path = ".".join(str(part) for part in error["loc"])
    prefix = f"{path}: " if path else ""
    return f"{prefix}{error['msg']}"


@app.websocket("/ws/document-preparation")
async def comparison_socket(websocket: WebSocket) -> None:
    """Accept replacement runs and stream each pair's two visible stages."""

    await websocket.accept()
    active: asyncio.Task[None] | None = None
    send_lock = asyncio.Lock()

    async def emit(event: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(event)

    try:
        while True:
            payload = await websocket.receive_json()
            if active is not None:
                if not active.done():
                    active.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await active

            try:
                request = ComparisonRequest.model_validate(payload)
                service.validate_registry(request)
            except (ValidationError, ValueError) as exc:
                message = (
                    _validation_message(exc)
                    if isinstance(exc, ValidationError)
                    else str(exc)
                )
                await emit(
                    {
                        "type": "run.failed",
                        "run_id": payload.get("run_id")
                        if isinstance(payload, dict)
                        else None,
                        "message": message,
                    }
                )
                continue

            active = asyncio.create_task(service.compare(request, emit))
    except WebSocketDisconnect:
        pass
    finally:
        if active is not None:
            if not active.done():
                active.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await active
