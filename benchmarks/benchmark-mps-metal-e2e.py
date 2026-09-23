#!/usr/bin/env python3
"""End-to-end A/B benchmark for fused Parakeet residual-add + LayerNorm on MPS.

The Metal implementation and Kestrel monkey patch live only in this process.
No installed package or production setting is changed.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Iterator
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")
KESTREL_VERSION = "0.8.0"
KERNELS_VERSION = "0.7.0"
WIDTH = 1024
GROUP_SIZE = 256

SHADER = r"""
#include <metal_stdlib>
using namespace metal;

kernel void add_scaled_layer_norm_1024(
    device half* total [[buffer(0)]],
    device half* normalized [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device const half* update [[buffer(3)]],
    device const half* weight [[buffer(4)]],
    device const half* bias [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial_sum[8];
  threadgroup float partial_square[8];
  threadgroup float stats[2];
  const uint base = row * 1024;
  float sum = 0.0f;
  float square = 0.0f;

  for (uint channel = lane; channel < 1024; channel += 256) {
    const uint index = base + channel;
    half value = residual[index] + half(alpha) * update[index];
    total[index] = value;
    const float value_f = float(value);
    sum += value_f;
    square += value_f * value_f;
  }
  sum = simd_sum(sum);
  square = simd_sum(square);
  if (simd_lane == 0) {
    partial_sum[simd_group] = sum;
    partial_square[simd_group] = square;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group == 0) {
    float group_sum = simd_lane < 8 ? partial_sum[simd_lane] : 0.0f;
    float group_square = simd_lane < 8 ? partial_square[simd_lane] : 0.0f;
    group_sum = simd_sum(group_sum);
    group_square = simd_sum(group_square);
    if (simd_lane == 0) {
      const float mean = group_sum * (1.0f / 1024.0f);
      const float variance = max(
          group_square * (1.0f / 1024.0f) - mean * mean, 0.0f);
      stats[0] = mean;
      stats[1] = rsqrt(variance + 1.0e-5f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float mean = stats[0];
  const float inverse_std = stats[1];
  for (uint channel = lane; channel < 1024; channel += 256) {
    const uint index = base + channel;
    normalized[index] = half(
        (float(total[index]) - mean) * inverse_std * float(weight[channel])
        + float(bias[channel]));
  }
}
"""


@contextlib.contextmanager
def exclusive_mps_lock() -> Iterator[None]:
    """Hold a process-owned atomic lock for the complete model run."""
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        try:
            owner = LOCK_PATH.read_text(errors="replace").strip()
        except (IsADirectoryError, OSError):
            owner = "owned by a directory-lock benchmark"
        raise RuntimeError(f"MPS benchmark lock exists: {LOCK_PATH} ({owner})") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            LOCK_PATH.unlink()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM WAV")
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def default_wavs() -> list[Path]:
    candidates = [Path("assets/demo-input.wav")]
    for directory in (Path("assets/samples"), Path("samples"), Path("tests/samples")):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.wav")))
    return list(dict.fromkeys(path for path in candidates if path.is_file()))


def build_fused_runtime(torch: Any) -> tuple[Any, Callable[..., Any]]:
    """Return a replacement conformer table and its guarded fused operation."""
    from kestrel_kernels import get_runtime

    runtime = get_runtime()
    original = runtime.conformer
    library = torch.mps.compile_shader(SHADER)
    kernel = library.add_scaled_layer_norm_1024
    if kernel.thread_execution_width != 32 or kernel.max_threads_per_threadgroup < GROUP_SIZE:
        raise RuntimeError("Metal kernel needs a 32-lane SIMD group and 256-thread groups")

    def fused(
        residual: Any,
        update: Any,
        alpha: float,
        weight: Any,
        bias: Any,
        eps: float = 1e-5,
    ) -> tuple[Any, Any]:
        supported = (
            residual.device.type == "mps"
            and residual.dtype == torch.float16
            and update.device == residual.device
            and update.dtype == residual.dtype
            and weight.device == residual.device
            and bias.device == residual.device
            and weight.dtype == residual.dtype
            and bias.dtype == residual.dtype
            and residual.is_contiguous()
            and update.is_contiguous()
            and weight.is_contiguous()
            and bias.is_contiguous()
            and residual.shape == update.shape
            and residual.ndim >= 2
            and residual.shape[-1] == WIDTH
            and weight.shape == (WIDTH,)
            and bias.shape == (WIDTH,)
            and float(eps) == 1.0e-5
            and float(alpha) in (0.5, 1.0)
        )
        if not supported:
            return original.add_scaled_layer_norm(
                residual, update, alpha, weight, bias, eps
            )
        rows = residual.numel() // WIDTH
        total = torch.empty_like(residual)
        normalized = torch.empty_like(residual)
        kernel(
            total,
            normalized,
            residual,
            update,
            weight,
            bias,
            float(alpha),
            threads=GROUP_SIZE * rows,
            group_size=GROUP_SIZE,
        )
        return total, normalized

    return replace(original, add_scaled_layer_norm=fused), fused


@contextlib.contextmanager
def installed_conformer(table: Any) -> Iterator[None]:
    """Replace the cached runtime table, then restore it even after failure."""
    from kestrel_kernels import get_runtime

    runtime = get_runtime()
    had_cached = "conformer" in runtime.__dict__
    original = runtime.__dict__.get("conformer")
    runtime.__dict__["conformer"] = table
    try:
        yield
    finally:
        if had_cached:
            runtime.__dict__["conformer"] = original
        else:
            runtime.__dict__.pop("conformer", None)


@contextlib.contextmanager
def recorded_tokens() -> Iterator[list[list[int]]]:
    """Record every Kestrel tokenizer decode call made by one transcription."""
    from kestrel.models.parakeet_tdt.tokenizer import ParakeetTokenizer

    calls: list[list[int]] = []
    original = ParakeetTokenizer.decode

    def decode(self: Any, token_ids: list[int]) -> str:
        calls.append([int(token) for token in token_ids])
        return original(self, token_ids)

    ParakeetTokenizer.decode = decode
    try:
        yield calls
    finally:
        ParakeetTokenizer.decode = original


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def run_exact(speech: Any, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    with recorded_tokens() as token_calls:
        started = time.perf_counter()
        result = speech.transcribe(
            audio=audio, sample_rate=sample_rate, timestamps="none"
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "latency_ms": elapsed_ms,
        "text": normalize_text(result.get("text", "")),
        "token_calls": token_calls,
        "final_tokens": token_calls[-1] if token_calls else [],
    }


def run_streaming(
    speech: Any,
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float,
) -> dict[str, Any]:
    frame_count = max(1, round(sample_rate * frame_ms / 1000.0))

    async def chunks():
        for offset in range(0, len(audio), frame_count):
            yield audio[offset : offset + frame_count]

    snapshots: list[str] = []
    with recorded_tokens() as token_calls:
        started = time.perf_counter()
        stream = speech.transcribe(
            audio=chunks(), sample_rate=sample_rate, timestamps="none", stream=True
        )
        for update in stream:
            snapshots.append(normalize_text(update.get("text", "")))
        result = stream.result()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "latency_ms": elapsed_ms,
        "text": normalize_text(result.get("text", "")),
        "token_calls": token_calls,
        "final_tokens": token_calls[-1] if token_calls else [],
        "snapshots": snapshots,
    }


def compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "latency_ms"}


def median(values: list[float]) -> float:
    return round(statistics.median(values), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="*", type=Path, help="WAVs; default: demo plus sample directories")
    parser.add_argument("--rounds", type=int, default=3, help="warm timed rounds per arm and mode")
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--tail-ms", type=float, default=320.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.frame_ms <= 0 or args.tail_ms < 0:
        parser.error("--rounds and --frame-ms must be positive; --tail-ms must be non-negative")
    wavs = args.wav or default_wavs()
    if not wavs:
        parser.error("no WAV inputs found")

    versions = {
        "kestrel": version("kestrel"),
        "kestrel-kernels": version("kestrel-kernels"),
        "torch": version("torch"),
    }
    if versions["kestrel"] != KESTREL_VERSION or versions["kestrel-kernels"] != KERNELS_VERSION:
        raise RuntimeError(
            "monkey patch refused: expected kestrel 0.8.0 and kestrel-kernels 0.7.0, "
            f"got {versions['kestrel']} and {versions['kestrel-kernels']}"
        )

    with exclusive_mps_lock():
        import torch
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("this benchmark requires macOS on arm64")
        if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("Apple MPS with torch.mps.compile_shader is required")

        from kestrel_kernels import get_runtime
        from ptt_worker import load_model

        photon = None
        try:
            photon, speech = load_model("mps", announce=False)
            original_table = get_runtime().conformer
            fused_table, _fused = build_fused_runtime(torch)
            inputs = []
            for path in wavs:
                audio, sample_rate = read_wav(path)
                tail = np.zeros(round(sample_rate * args.tail_ms / 1000.0), dtype=np.float32)
                audio_with_tail = np.concatenate((audio, tail))
                modes = (
                    ("exact", lambda: run_exact(speech, audio_with_tail, sample_rate)),
                    ("streaming", lambda: run_streaming(
                        speech, audio_with_tail, sample_rate, frame_ms=args.frame_ms
                    )),
                )
                mode_reports = []
                for mode, invoke in modes:
                    with installed_conformer(original_table):
                        baseline_parity = invoke()
                    with installed_conformer(fused_table):
                        fused_parity = invoke()

                    timings: dict[str, list[float]] = {"baseline": [], "fused": []}
                    for round_index in range(args.rounds):
                        order = ("baseline", "fused") if round_index % 2 == 0 else ("fused", "baseline")
                        for arm in order:
                            table = original_table if arm == "baseline" else fused_table
                            with installed_conformer(table):
                                timings[arm].append(float(invoke()["latency_ms"]))
                    baseline_ms = median(timings["baseline"])
                    fused_ms = median(timings["fused"])
                    mode_reports.append({
                        "mode": mode,
                        "baseline": compact_run(baseline_parity),
                        "fused": compact_run(fused_parity),
                        "transcript_exact": baseline_parity["text"] == fused_parity["text"],
                        "final_tokens_exact": baseline_parity["final_tokens"] == fused_parity["final_tokens"],
                        "token_trace_exact": baseline_parity["token_calls"] == fused_parity["token_calls"],
                        "warm_latency_ms": {
                            "baseline_samples": [round(value, 2) for value in timings["baseline"]],
                            "fused_samples": [round(value, 2) for value in timings["fused"]],
                            "baseline_median": baseline_ms,
                            "fused_median": fused_ms,
                            "speedup": round(baseline_ms / fused_ms, 4),
                        },
                    })
                inputs.append({
                    "wav": str(path),
                    "audio_seconds": round(len(audio) / sample_rate, 6),
                    "sample_rate": sample_rate,
                    "modes": mode_reports,
                })
        finally:
            if photon is not None:
                with contextlib.redirect_stdout(None):
                    photon.__exit__(None, None, None)

    all_modes = [mode for item in inputs for mode in item["modes"]]
    baseline_total = sum(mode["warm_latency_ms"]["baseline_median"] for mode in all_modes)
    fused_total = sum(mode["warm_latency_ms"]["fused_median"] for mode in all_modes)
    report = {
        "experiment": "parakeet-mps-fused-add-layernorm-e2e",
        "versions": versions,
        "platform": platform.platform(),
        "patch_scope": "process-local; exact-version guarded; production unchanged",
        "rounds": args.rounds,
        "frame_ms": args.frame_ms,
        "tail_ms": args.tail_ms,
        "inputs": inputs,
        "all_transcripts_exact": all(mode["transcript_exact"] for mode in all_modes),
        "all_final_tokens_exact": all(mode["final_tokens_exact"] for mode in all_modes),
        "all_token_traces_exact": all(mode["token_trace_exact"] for mode in all_modes),
        "total_warm_latency_ms": {
            "baseline_sum_of_medians": round(baseline_total, 2),
            "fused_sum_of_medians": round(fused_total, 2),
            "speedup": round(baseline_total / fused_total, 4),
        },
        "notes": [
            "Streaming audio is fed without real-time sleeps, so latency measures warm inference work.",
            "Parity runs are separate from timed rounds; timed arms alternate order.",
            "final_tokens is the final cumulative tokenizer decode; token_trace includes provisional streaming decodes.",
        ],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["all_transcripts_exact"] and report["all_final_tokens_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
