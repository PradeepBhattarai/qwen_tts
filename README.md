# Qwen3 TTS Service

Standalone serving layer around the local `Qwen3-TTS-streaming` checkout.

This repo now contains:

- `qwen_tts_services/`: backend service package
- `app.py`: FastAPI entrypoint
- `frontend/index.html`: browser client for websocket streaming tests
- `frontend/server.py`: tiny static file server

## Install

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -r requirements.txt
```

This creates a local `.venv`, installs the backend dependencies, and installs the local `Qwen3-TTS-streaming` checkout in editable mode.

If you need GPU-specific Torch / FlashAttention wheels, install those with `uv pip install ...` first based on the instructions in `Qwen3-TTS-streaming/README.md`, then run the command above.

## Run The Backend

Plain HTTP and WS:

```bash
uv run python serve_backend.py
```

TLS for HTTPS and WSS:

```bash
QWEN_TTS_PORT=8443 \
QWEN_TTS_SSL_CERTFILE=/path/to/cert.pem \
QWEN_TTS_SSL_KEYFILE=/path/to/key.pem \
uv run python serve_backend.py
```

Useful environment variables:

- `QWEN_TTS_MODEL_NAME`
- `QWEN_TTS_DEVICE_MAP`
- `QWEN_TTS_DTYPE`
- `QWEN_TTS_ATTN_IMPLEMENTATION`
- `QWEN_TTS_SOURCE_DIR`
- `QWEN_TTS_HOST`
- `QWEN_TTS_PORT`
- `QWEN_TTS_PRELOAD_MODEL`
- `QWEN_TTS_ENABLE_STREAMING_OPTIMIZATIONS`

The backend currently targets the Base voice-clone model flow, because the upstream streaming API is exposed through `stream_generate_voice_clone(...)`.

## Run The Frontend

```bash
uv run python frontend/server.py
```

Open `http://127.0.0.1:8080`.

## HTTP API

`POST /api/v1/tts`

JSON body example:

```json
{
  "text": "Hello from the non-streaming endpoint.",
  "language": "Auto",
  "reference_audio_path": "/absolute/path/to/reference.wav",
  "reference_text": "Reference transcript for in-context cloning."
}
```

The response body is `audio/wav`.

## WebSocket API

`WS /api/v1/tts/stream`

1. Connect.
2. Send one JSON request with the same voice-clone fields as the HTTP API plus streaming controls:
   - `emit_every_frames`
   - `decode_window_frames`
   - `max_frames`
3. Receive:
   - a JSON `stream_start` frame
   - binary `pcm_s16le` mono audio chunks
   - a JSON `stream_end` frame

Example websocket request:

```json
{
  "text": "Hello from the streaming endpoint.",
  "language": "Auto",
  "reference_audio_path": "/absolute/path/to/reference.wav",
  "reference_text": "Reference transcript for in-context cloning.",
  "emit_every_frames": 4,
  "decode_window_frames": 80,
  "max_frames": 10000
}
```
