from __future__ import annotations

import sys
from pathlib import Path
from typing import Type


def _prepend_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def import_qwen_tts_model(source_dir: Path) -> Type[object]:
    """
    Prefer the local streaming fork if it exists, then import the model wrapper.
    """
    resolved_dir = source_dir.expanduser().resolve()
    if resolved_dir.exists():
        _prepend_path(resolved_dir)

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import `qwen_tts`. Install `-e ./Qwen3-TTS-streaming` or "
            "point QWEN_TTS_SOURCE_DIR at a checkout of the streaming fork."
        ) from exc

    if not hasattr(Qwen3TTSModel, "stream_generate_voice_clone"):
        raise RuntimeError(
            "The imported `qwen_tts` package does not expose `stream_generate_voice_clone`. "
            "Use the streaming fork in `Qwen3-TTS-streaming`."
        )

    return Qwen3TTSModel
