from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VoiceCloneRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    reference_audio_path: str | None = None
    reference_audio_url: str | None = None
    reference_audio_base64: str | None = None
    reference_text: str | None = None
    x_vector_only_mode: bool = False
    non_streaming_mode: bool = False
    do_sample: bool | None = None
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    temperature: float | None = Field(default=None, gt=0.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    subtalker_dosample: bool | None = None
    subtalker_top_k: int | None = Field(default=None, ge=1)
    subtalker_top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    subtalker_temperature: float | None = Field(default=None, gt=0.0)
    max_new_tokens: int | None = Field(default=None, ge=1)

    def generation_kwargs(self) -> dict[str, Any]:
        names = (
            "do_sample",
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "subtalker_dosample",
            "subtalker_top_k",
            "subtalker_top_p",
            "subtalker_temperature",
            "max_new_tokens",
        )
        kwargs: dict[str, Any] = {}
        for name in names:
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs


class StreamingVoiceCloneRequest(VoiceCloneRequest):
    emit_every_frames: int = Field(default=4, ge=1)
    decode_window_frames: int = Field(default=80, ge=1)
    overlap_samples: int = Field(default=0, ge=0)
    max_frames: int = Field(default=10000, ge=1)
    use_optimized_decode: bool = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device_map: str
    dtype: str
    attn_implementation: str
    upstream_source_dir: str
    upstream_source_exists: bool
    tts_model_type: str | None = None
    tokenizer_type: str | None = None
