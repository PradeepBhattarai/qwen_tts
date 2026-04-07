from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from .bootstrap import import_qwen_tts_model
from .config import Settings
from .schemas import HealthResponse, StreamingVoiceCloneRequest, VoiceCloneRequest


@dataclass(slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int


@dataclass(slots=True)
class StreamChunk:
    audio: np.ndarray
    sample_rate: int


class QwenTTSService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def status(self) -> HealthResponse:
        base_model = getattr(self._model, "model", None) if self._model is not None else None
        return HealthResponse(
            status="ok",
            model_loaded=self._model is not None,
            model_name=self.settings.model_name,
            device_map=self.settings.device_map,
            dtype=self.settings.dtype,
            attn_implementation=self.settings.attn_implementation,
            upstream_source_dir=str(self.settings.upstream_source_dir),
            upstream_source_exists=self.settings.upstream_source_dir.exists(),
            tts_model_type=getattr(base_model, "tts_model_type", None),
            tokenizer_type=getattr(base_model, "tokenizer_type", None),
        )

    def ensure_model(self):
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            model_cls = import_qwen_tts_model(self.settings.upstream_source_dir)
            import torch

            torch.set_float32_matmul_precision("high")

            dtype = self._resolve_dtype(torch, self.settings.dtype)
            model = model_cls.from_pretrained(
                self.settings.model_name,
                device_map=self.settings.device_map,
                dtype=dtype,
                attn_implementation=self.settings.attn_implementation,
            )

            if self.settings.enable_streaming_optimizations:
                model.enable_streaming_optimizations(
                    decode_window_frames=self.settings.stream_decode_window_frames,
                    use_compile=True,
                    use_cuda_graphs=self.settings.stream_use_cuda_graphs,
                    compile_mode=self.settings.stream_compile_mode,
                    use_fast_codebook=self.settings.stream_use_fast_codebook,
                    compile_codebook_predictor=self.settings.stream_compile_codebook_predictor,
                    compile_talker=self.settings.stream_compile_talker,
                )

            self._model = model
            return self._model

    def preload(self) -> None:
        self.ensure_model()

    def synthesize(self, request: VoiceCloneRequest) -> SynthesisResult:
        model = self.ensure_model()
        ref_audio = self._resolve_reference_audio(request)
        text = self._clean_text(request.text, field_name="text")
        ref_text = self._clean_optional_text(request.reference_text)
        kwargs = request.generation_kwargs()

        with self._generation_lock:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=request.language or "Auto",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=request.x_vector_only_mode,
                non_streaming_mode=request.non_streaming_mode,
                **kwargs,
            )

        if not wavs:
            raise RuntimeError("The model completed without returning audio.")

        return SynthesisResult(audio=np.asarray(wavs[0], dtype=np.float32), sample_rate=int(sample_rate))

    def stream_synthesize(
        self, request: StreamingVoiceCloneRequest
    ) -> Generator[StreamChunk, None, None]:
        model = self.ensure_model()
        ref_audio = self._resolve_reference_audio(request)
        text = self._clean_text(request.text, field_name="text")
        ref_text = self._clean_optional_text(request.reference_text)
        kwargs = request.generation_kwargs()

        with self._generation_lock:
            for chunk, sample_rate in model.stream_generate_voice_clone(
                text=text,
                language=request.language or "Auto",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=request.x_vector_only_mode,
                non_streaming_mode=request.non_streaming_mode,
                emit_every_frames=request.emit_every_frames,
                decode_window_frames=request.decode_window_frames,
                overlap_samples=request.overlap_samples,
                max_frames=request.max_frames,
                use_optimized_decode=request.use_optimized_decode,
                **kwargs,
            ):
                yield StreamChunk(audio=np.asarray(chunk, dtype=np.float32), sample_rate=int(sample_rate))

    @staticmethod
    def _resolve_dtype(torch_module, dtype_name: str):
        name = dtype_name.strip().lower()
        mapping = {
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "half": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        try:
            return mapping[name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype `{dtype_name}`. Use one of: {', '.join(sorted(mapping))}."
            ) from exc

    def _resolve_reference_audio(self, request: VoiceCloneRequest) -> str:
        ref_path = self._clean_optional_text(request.reference_audio_path)
        ref_url = self._clean_optional_text(request.reference_audio_url)
        ref_b64 = self._clean_optional_text(request.reference_audio_base64)

        provided = [value for value in (ref_path, ref_url, ref_b64) if value is not None]
        if len(provided) != 1:
            raise ValueError(
                "Provide exactly one reference audio source: "
                "`reference_audio_path`, `reference_audio_url`, or `reference_audio_base64`."
            )

        if not request.x_vector_only_mode and not self._clean_optional_text(request.reference_text):
            raise ValueError("`reference_text` is required unless `x_vector_only_mode` is true.")

        if ref_path is not None:
            ref_path_obj = Path(ref_path).expanduser()
            if not ref_path_obj.exists():
                raise ValueError(f"Reference audio path does not exist: {ref_path_obj}")
            return str(ref_path_obj)
        if ref_url is not None:
            return ref_url
        assert ref_b64 is not None
        return ref_b64

    @staticmethod
    def _clean_text(value: str, *, field_name: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"`{field_name}` must not be empty.")
        return cleaned

    @staticmethod
    def _clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None
