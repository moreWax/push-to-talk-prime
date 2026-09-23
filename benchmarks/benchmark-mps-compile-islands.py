#!/usr/bin/env python3
"""Compare eager, whole-encoder compile, and safe pure-Torch MPS islands.

This is a process-local experiment. It does not edit Kestrel or the production
worker. Native Kestrel ``*_into`` pybind calls stay outside Dynamo because their
mutation/alias contracts are not registered with PyTorch.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import wave
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_VERSIONS = {"kestrel": "0.8.0", "kestrel-kernels": "0.7.0", "torch": "2.14.0"}
DEFAULT_BUCKETS = (48, 80, 128, 224)
LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")


def acquire_lock() -> Path:
    try:
        descriptor = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        detail = "owner unknown"
        if LOCK_PATH.is_file():
            detail = LOCK_PATH.read_text(errors="replace").strip() or detail
        raise RuntimeError(f"another MPS benchmark holds {LOCK_PATH}: {detail}") from exc
    try:
        with os.fdopen(descriptor, "w") as owner:
            owner.write(json.dumps({"pid": os.getpid(), "script": str(Path(__file__).resolve())}) + "\n")
    except Exception:
        LOCK_PATH.unlink(missing_ok=True)
        raise
    return LOCK_PATH


def release_lock(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def check_environment(torch: Any) -> dict[str, str]:
    actual = {name: version(name) for name in EXPECTED_VERSIONS}
    bad = [f"{name}=={actual[name]} (expected {want})" for name, want in EXPECTED_VERSIONS.items() if actual[name] != want]
    if bad:
        raise RuntimeError("unsupported experiment versions: " + ", ".join(bad))
    if sys.platform != "darwin" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
        raise RuntimeError("this experiment requires Apple MPS on arm64 macOS")
    return actual


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("input must be mono 16-bit PCM WAV")
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def counters(torch: Any) -> dict[str, dict[str, int]]:
    names = ("frames", "stats", "graph_break", "inductor", "aot_autograd")
    return {name: {str(k): int(v) for k, v in torch._dynamo.utils.counters[name].items()} for name in names if torch._dynamo.utils.counters.get(name)}


def normalize_buckets(values: list[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(x <= 0 for x in result) or any(a >= b for a, b in zip(result, result[1:])):
        raise ValueError("--buckets must be strictly increasing positive integers")
    return result


class Experiment:
    def __init__(self, torch: Any, original: Any, buckets: tuple[int, ...]) -> None:
        self.torch = torch
        self.original = original
        self.buckets = buckets
        self.mode = "eager"
        self.whole: dict[int, Callable[..., Any]] = {}
        self.calls: list[dict[str, object]] = []
        self.fallbacks: Counter[str] = Counter()
        self.errors: list[str] = []

    def start(self, mode: str) -> None:
        self.mode, self.calls = mode, []

    def whole_compiled(self, model: Any, batch: int) -> Callable[..., Any]:
        if batch not in self.whole:
            self.whole[batch] = self.torch.compile(model.encode_subsampled, backend="inductor", fullgraph=False, dynamic=False)
        return self.whole[batch]

    def encode(self, model: Any, features: Any, mask: Any) -> tuple[Any, Any]:
        self.torch.mps.synchronize()
        started = time.perf_counter()
        bucket = None
        fallback = None
        try:
            if self.mode == "eager":
                result = self.original(model, features, mask)
            else:
                factor = model.config.encoder.subsampling_factor
                length = (features.shape[1] + factor - 1) // factor
                bucket = next((size for size in self.buckets if size >= length), None)
                if bucket is None:
                    fallback = "over_max_bucket"
                    result = self.original(model, features, mask)
                else:
                    hidden, valid = model.encoder.subsampling(features, mask)
                    hidden = self.torch.nn.functional.pad(hidden, (0, 0, 0, bucket - hidden.shape[1])).contiguous()
                    valid = self.torch.nn.functional.pad(valid, (0, bucket - valid.shape[1])).contiguous()
                    try:
                        fn = self.whole_compiled(model, int(features.shape[0])) if self.mode == "previous_compile" else model.encode_subsampled
                        encoded, encoded_valid = fn(hidden, valid)
                        result = encoded[:, :length], encoded_valid[:, :length]
                    except Exception as exc:
                        fallback = "compile_error"
                        message = f"{type(exc).__name__}: {exc}"
                        if message not in self.errors:
                            self.errors.append(message)
                        result = self.original(model, features, mask)
                if fallback:
                    self.fallbacks[f"{self.mode}:{fallback}"] += 1
            self.torch.mps.synchronize()
            return result
        finally:
            self.calls.append({"mode": self.mode, "input_feature_frames": int(features.shape[1]), "batch": int(features.shape[0]), "bucket": bucket, "fallback": fallback, "latency_ms": round((time.perf_counter() - started) * 1000, 3)})


def install_islands(torch: Any, runtime: Any) -> Any:
    """Install only the add+LayerNorm island; keep native mutation opaque.

    Registering the two pybind ``*_into`` kernels as custom ops would require an
    exact schema, fake implementation, mutation declaration, and alias contract.
    Kestrel 0.7 exposes none. Guessing those contracts is unsafe, so disable is
    the experiment's intentional stop boundary.
    """
    original = runtime.conformer

    def add_norm(residual: Any, x: Any, alpha: float, weight: Any, bias: Any, eps: float) -> tuple[Any, Any]:
        total = torch.add(residual, x, alpha=alpha)
        return total, torch.nn.functional.layer_norm(total, (total.shape[-1],), weight, bias, eps)

    compiled_add_norm = torch.compile(add_norm, backend="inductor", fullgraph=True, dynamic=False)
    opaque = lambda fn: torch.compiler.disable(fn, recursive=True)
    # These MPS functions ultimately call raw pybind ``*_into(out, ...)`` ops.
    runtime.__dict__["conformer"] = replace(
        original,
        add_scaled_layer_norm=compiled_add_norm,
        depthwise_conv_bn_silu=opaque(original.depthwise_conv_bn_silu),
        rel_attention=opaque(original.rel_attention),
    )
    return original


def run(speech: Any, experiment: Experiment, audio: np.ndarray, rate: int, mode: str) -> dict[str, object]:
    from ptt_worker import normalize_text
    experiment.start(mode)
    started = time.perf_counter()
    result = speech.transcribe(audio=audio, sample_rate=rate, timestamps="none")
    elapsed = (time.perf_counter() - started) * 1000
    return {"mode": mode, "total_ms": round(elapsed, 3), "text": normalize_text(str(result.get("text", ""))), "encoder_calls": list(experiment.calls), "encoder_total_ms": round(sum(float(x["latency_ms"]) for x in experiment.calls), 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path, default=Path("assets/demo-input.wav"))
    parser.add_argument("--eager-rounds", type=int, default=3)
    parser.add_argument("--hot-rounds", type=int, default=3)
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.eager_rounds < 1 or args.hot_rounds < 1:
        parser.error("round counts must be positive")
    try:
        buckets = normalize_buckets(args.buckets)
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    versions = check_environment(torch)
    audio, rate = read_wav(args.wav)
    from kestrel_kernels import get_runtime
    from kestrel.models.parakeet_tdt.model import ParakeetTdt
    from ptt_worker import load_model

    original_encode = ParakeetTdt.encode
    experiment = Experiment(torch, original_encode, buckets)
    ParakeetTdt.encode = lambda model, features, mask: experiment.encode(model, features, mask)
    photon = None
    lock = acquire_lock()
    original_conformer = None
    try:
        photon, speech = load_model("mps", announce=False)
        eager = [run(speech, experiment, audio, rate, "eager") for _ in range(args.eager_rounds)]

        torch._dynamo.reset(); torch._dynamo.utils.counters.clear()
        previous_cold = run(speech, experiment, audio, rate, "previous_compile")
        previous_hot = [run(speech, experiment, audio, rate, "previous_compile") for _ in range(args.hot_rounds)]
        previous_counters = counters(torch)

        torch._dynamo.reset(); torch._dynamo.utils.counters.clear()
        runtime = get_runtime()
        original_conformer = install_islands(torch, runtime)
        islands_cold = run(speech, experiment, audio, rate, "islands")
        islands_hot = [run(speech, experiment, audio, rate, "islands") for _ in range(args.hot_rounds)]
        islands_counters = counters(torch)
    finally:
        ParakeetTdt.encode = original_encode
        if original_conformer is not None:
            get_runtime().__dict__["conformer"] = original_conformer
        if photon is not None:
            with contextlib.redirect_stdout(None):
                photon.__exit__(None, None, None)
        release_lock(lock)

    baseline = eager[-1]["text"]
    compared = [previous_cold, *previous_hot, islands_cold, *islands_hot]
    equality = {"previous_cold": previous_cold["text"] == baseline, "previous_hot": [x["text"] == baseline for x in previous_hot], "islands_cold": islands_cold["text"] == baseline, "islands_hot": [x["text"] == baseline for x in islands_hot]}
    equality["all"] = all(x["text"] == baseline for x in compared)
    median = lambda rows: round(statistics.median(float(x["encoder_total_ms"]) for x in rows), 3)
    hot_latency = {"eager_median_encoder_ms": median(eager), "previous_compile_median_encoder_ms": median(previous_hot), "islands_median_encoder_ms": median(islands_hot)}
    hot_latency["islands_vs_eager_ratio"] = round(hot_latency["islands_median_encoder_ms"] / hot_latency["eager_median_encoder_ms"], 4)
    hot_latency["islands_vs_previous_ratio"] = round(hot_latency["islands_median_encoder_ms"] / hot_latency["previous_compile_median_encoder_ms"], 4)
    graph_summary = {
        "eager_unique_graphs": 0,
        "previous_compile_unique_graphs": previous_counters.get("stats", {}).get("unique_graphs", 0),
        "previous_compile_graph_breaks": sum(previous_counters.get("graph_break", {}).values()),
        "islands_unique_graphs": islands_counters.get("stats", {}).get("unique_graphs", 0),
        "islands_graph_breaks": sum(islands_counters.get("graph_break", {}).values()),
    }
    report = {
        "experiment": "apple-mps-safe-compile-islands", "versions": versions, "platform": platform.platform(),
        "wav": str(args.wav), "audio_seconds": round(len(audio) / rate, 6), "sample_rate": rate, "buckets": buckets,
        "boundary_decision": {"custom_op_registration": "stopped_unsafe", "reason": "Kestrel raw pybind *_into kernels do not publish PyTorch mutation/alias schemas or fake implementations; guessed custom-op contracts could miscompile.", "opaque_kernels": ["ternary_gemm_into", "conformer_depthwise_bn_silu_into", "conformer_rel_attn_into"], "compiled_island": "torch.add + torch.nn.functional.layer_norm"},
        "eager_runs": eager, "previous_compile": {"cold": previous_cold, "hot": previous_hot, "dynamo_counters": previous_counters},
        "islands": {"cold": islands_cold, "hot": islands_hot, "dynamo_counters": islands_counters},
        "graph_summary": graph_summary, "hot_encoder_latency": hot_latency, "transcript_equality": equality, "path_fallback_counts": dict(experiment.fallbacks), "compile_errors": experiment.errors,
        "benchmark_lock": str(LOCK_PATH), "warning": "Private process-local experiment only; production and installed .venv sources are unchanged."
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0 if equality["all"] and not experiment.fallbacks else 2


if __name__ == "__main__":
    raise SystemExit(main())
