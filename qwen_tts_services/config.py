from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True, slots=True)
class Settings:
    model_name: str
    device_map: str
    dtype: str
    attn_implementation: str
    upstream_source_dir: Path
    preload_model: bool
    cors_allow_origins: tuple[str, ...]
    enable_streaming_optimizations: bool
    stream_emit_every_frames: int
    stream_decode_window_frames: int
    stream_overlap_samples: int
    stream_max_frames: int
    stream_compile_mode: str
    stream_use_cuda_graphs: bool
    stream_use_fast_codebook: bool
    stream_compile_codebook_predictor: bool
    stream_compile_talker: bool
    host: str
    port: int
    ssl_certfile: str | None
    ssl_keyfile: str | None
    frontend_host: str
    frontend_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parent.parent
        source_dir = Path(
            os.getenv("QWEN_TTS_SOURCE_DIR", str(repo_root / "Qwen3-TTS-streaming"))
        ).expanduser()
        origins_raw = os.getenv("QWEN_TTS_CORS_ALLOW_ORIGINS", "*")
        origins = tuple(item.strip() for item in origins_raw.split(",") if item.strip()) or ("*",)

        return cls(
            model_name=_env_str("QWEN_TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            device_map=_env_str("QWEN_TTS_DEVICE_MAP", "cuda:0"),
            dtype=_env_str("QWEN_TTS_DTYPE", "bfloat16"),
            attn_implementation=_env_str("QWEN_TTS_ATTN_IMPLEMENTATION", "flash_attention_2"),
            upstream_source_dir=source_dir,
            preload_model=_env_bool("QWEN_TTS_PRELOAD_MODEL", False),
            cors_allow_origins=origins,
            enable_streaming_optimizations=_env_bool("QWEN_TTS_ENABLE_STREAMING_OPTIMIZATIONS", True),
            stream_emit_every_frames=_env_int("QWEN_TTS_STREAM_EMIT_EVERY_FRAMES", 4),
            stream_decode_window_frames=_env_int("QWEN_TTS_STREAM_DECODE_WINDOW_FRAMES", 80),
            stream_overlap_samples=_env_int("QWEN_TTS_STREAM_OVERLAP_SAMPLES", 0),
            stream_max_frames=_env_int("QWEN_TTS_STREAM_MAX_FRAMES", 10000),
            stream_compile_mode=_env_str("QWEN_TTS_STREAM_COMPILE_MODE", "reduce-overhead"),
            stream_use_cuda_graphs=_env_bool("QWEN_TTS_STREAM_USE_CUDA_GRAPHS", False),
            stream_use_fast_codebook=_env_bool("QWEN_TTS_STREAM_USE_FAST_CODEBOOK", True),
            stream_compile_codebook_predictor=_env_bool(
                "QWEN_TTS_STREAM_COMPILE_CODEBOOK_PREDICTOR", True
            ),
            stream_compile_talker=_env_bool("QWEN_TTS_STREAM_COMPILE_TALKER", True),
            host=_env_str("QWEN_TTS_HOST", "0.0.0.0"),
            port=_env_int("QWEN_TTS_PORT", 8000),
            ssl_certfile=os.getenv("QWEN_TTS_SSL_CERTFILE"),
            ssl_keyfile=os.getenv("QWEN_TTS_SSL_KEYFILE"),
            frontend_host=_env_str("FRONTEND_HOST", "127.0.0.1"),
            frontend_port=_env_int("FRONTEND_PORT", 8080),
        )
