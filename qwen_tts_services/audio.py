from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Clamp model audio into a mono float32 waveform."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(waveform, -1.0, 1.0)


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(len(audio)) / float(sample_rate)


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    waveform = normalize_audio(audio)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def pcm_s16le_bytes(audio: np.ndarray) -> bytes:
    waveform = normalize_audio(audio)
    return (waveform * 32767.0).astype(np.int16).tobytes()
