from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from .audio import duration_seconds, pcm_s16le_bytes, wav_bytes
from .config import Settings
from .schemas import HealthResponse, StreamingVoiceCloneRequest, VoiceCloneRequest
from .service import QwenTTSService


logger = logging.getLogger(__name__)


def _parse_stream_request(raw_payload: str) -> StreamingVoiceCloneRequest:
    validator = getattr(StreamingVoiceCloneRequest, "model_validate_json", None)
    if callable(validator):
        return validator(raw_payload)
    return StreamingVoiceCloneRequest.parse_raw(raw_payload)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    service = QwenTTSService(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if resolved_settings.preload_model:
            await run_in_threadpool(service.preload)
        yield

    app = FastAPI(
        title="Qwen3 TTS Service",
        version="0.1.0",
        description="HTTP and websocket serving layer for the local Qwen3-TTS streaming fork.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.tts_service = service

    allow_credentials = resolved_settings.cors_allow_origins != ("*",)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_allow_origins),
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "qwen3-tts",
            "docs": "/docs",
            "health": "/api/v1/health",
            "http_tts": "/api/v1/tts",
            "ws_tts": "/api/v1/tts/stream",
        }

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return service.status()

    @app.post("/api/v1/tts")
    async def synthesize_tts(request: VoiceCloneRequest) -> Response:
        try:
            result = await run_in_threadpool(service.synthesize, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected synthesis failure")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        duration = duration_seconds(result.audio, result.sample_rate)
        return Response(
            content=wav_bytes(result.audio, result.sample_rate),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="tts.wav"',
                "X-Sample-Rate": str(result.sample_rate),
                "X-Audio-Duration-Sec": f"{duration:.3f}",
            },
        )

    @app.websocket("/api/v1/tts/stream")
    async def stream_tts(websocket: WebSocket) -> None:
        await websocket.accept()

        try:
            raw_payload = await websocket.receive_text()
            request = _parse_stream_request(raw_payload)
        except WebSocketDisconnect:
            return
        except ValidationError as exc:
            await websocket.send_json({"type": "error", "detail": "Invalid request", "errors": exc.errors()})
            await websocket.close(code=1003)
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1003)
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict | None, bytes | None]] = asyncio.Queue()
        stop_event = threading.Event()

        def publish(kind: str, payload: dict | None = None, data: bytes | None = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload, data))

        def worker() -> None:
            started = False
            chunk_count = 0
            started_at = time.perf_counter()

            try:
                for chunk in service.stream_synthesize(request):
                    if stop_event.is_set():
                        break

                    if not started:
                        publish(
                            "json",
                            {
                                "type": "stream_start",
                                "sample_rate": chunk.sample_rate,
                                "channels": 1,
                                "encoding": "pcm_s16le",
                                "emit_every_frames": request.emit_every_frames,
                                "decode_window_frames": request.decode_window_frames,
                            },
                        )
                        started = True

                    publish("bytes", data=pcm_s16le_bytes(chunk.audio))
                    chunk_count += 1

                publish(
                    "json",
                    {
                        "type": "stream_end",
                        "chunk_count": chunk_count,
                        "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    },
                )
            except Exception as exc:
                logger.exception("Streaming synthesis failure")
                publish("json", {"type": "error", "detail": str(exc)})
            finally:
                publish("done")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                kind, payload, data = await queue.get()

                if kind == "json":
                    assert payload is not None
                    await websocket.send_json(payload)
                    if payload.get("type") == "error":
                        await websocket.close(code=1011)
                        return
                elif kind == "bytes":
                    await websocket.send_bytes(data or b"")
                elif kind == "done":
                    break
        except WebSocketDisconnect:
            stop_event.set()
            return
        except Exception:
            stop_event.set()
            raise
        finally:
            stop_event.set()

        try:
            await websocket.close(code=1000)
        except RuntimeError:
            logger.debug("WebSocket was already closed by the client.")

    return app
