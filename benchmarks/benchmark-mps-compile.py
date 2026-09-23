#!/usr/bin/env python3
"""Benchmark an isolated fixed-shape torch.compile Parakeet encoder on Apple MPS.

This script patches Kestrel only in this process. It never edits the environment
or changes the worker's production path/defaults.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import wave
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_VERSIONS = {
    "kestrel": "0.8.0",
    "kestrel-kernels": "0.7.0",
    "torch": "2.14.0",
}
DEFAULT_BUCKETS = (48, 80, 128, 224)
LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")


def acquire_benchmark_lock() -> Path:
    try:
        LOCK_PATH.mkdir()
    except FileExistsError as exc:
        owner = LOCK_PATH / "owner.json"
        detail = owner.read_text().strip() if owner.exists() else "owner unknown"
        raise RuntimeError(
            f"another MPS benchmark holds {LOCK_PATH}: {detail}"
        ) from exc
    try:
        (LOCK_PATH / "owner.json").write_text(
            json.dumps({"pid": os.getpid(), "script": str(Path(__file__).resolve())})
            + "\n"
        )
    except Exception:
        LOCK_PATH.rmdir()
        raise
    return LOCK_PATH


def release_benchmark_lock(lock: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (lock / "owner.json").unlink()
    lock.rmdir()


def check_environment(torch: Any) -> dict[str, str]:
    actual = {name: version(name) for name in EXPECTED_VERSIONS}
    mismatches = [
        f"{name}=={actual[name]} (expected {expected})"
        for name, expected in EXPECTED_VERSIONS.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise RuntimeError("unsupported experiment versions: " + ", ".join(mismatches))
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("this experiment requires macOS on arm64")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available")
    return actual


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("input must be mono 16-bit PCM WAV")
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def normalize_buckets(values: list[int]) -> tuple[int, ...]:
    buckets = tuple(values)
    if not buckets or any(value <= 0 for value in buckets) or any(
        left >= right for left, right in zip(buckets, buckets[1:])
    ):
        raise ValueError("--buckets must be strictly increasing positive integers")
    return buckets


def counter_snapshot(torch: Any) -> dict[str, dict[str, int]]:
    counters = torch._dynamo.utils.counters
    names = ("frames", "stats", "graph_break", "inductor", "aot_autograd")
    return {
        name: {str(key): int(value) for key, value in counters[name].items()}
        for name in names
        if counters.get(name)
    }


class EncoderExperiment:
    """Process-local replacement for ParakeetTdt.encode."""

    def __init__(
        self,
        torch: Any,
        original: Any,
        *,
        buckets: tuple[int, ...],
        fullgraph: bool,
    ) -> None:
        self.torch = torch
        self.original = original
        self.buckets = buckets
        self.fullgraph = fullgraph
        self.mode = "eager"
        self.compiled: dict[int, Any] = {}
        self.encoder_calls: list[dict[str, object]] = []
        self.fallbacks: Counter[str] = Counter()
        self.compile_errors: list[str] = []

    def start_run(self, mode: str) -> None:
        self.mode = mode
        self.encoder_calls = []

    def _compiled_for(self, model: Any, batch: int) -> Any:
        compiled = self.compiled.get(batch)
        if compiled is None:
            # dynamic=False plus padded bucket inputs produces one specialization
            # for each (batch, bucket) pair actually encountered.
            compiled = self.torch.compile(
                model.encode_subsampled,
                backend="inductor",
                fullgraph=self.fullgraph,
                dynamic=False,
            )
            self.compiled[batch] = compiled
        return compiled

    def encode(self, model: Any, features: Any, mask: Any) -> tuple[Any, Any]:
        self.torch.mps.synchronize()
        started = time.perf_counter()
        bucket: int | None = None
        fallback: str | None = None
        try:
            if self.mode == "eager":
                result = self.original(model, features, mask)
            else:
                factor = model.config.encoder.subsampling_factor
                length = (features.shape[1] + factor - 1) // factor
                bucket = next((size for size in self.buckets if size >= length), None)
                if bucket is None:
                    fallback = "over_max_bucket"
                    self.fallbacks[fallback] += 1
                    result = self.original(model, features, mask)
                else:
                    hidden, valid = model.encoder.subsampling(features, mask)
                    hidden = self.torch.nn.functional.pad(
                        hidden, (0, 0, 0, bucket - hidden.shape[1])
                    ).contiguous()
                    valid = self.torch.nn.functional.pad(
                        valid, (0, bucket - valid.shape[1])
                    ).contiguous()
                    try:
                        encoded, encoded_valid = self._compiled_for(
                            model, int(features.shape[0])
                        )(hidden, valid)
                        result = encoded[:, :length], encoded_valid[:, :length]
                    except Exception as exc:  # experimental fallback is reported
                        fallback = "compile_error"
                        self.fallbacks[fallback] += 1
                        message = f"{type(exc).__name__}: {exc}"
                        if message not in self.compile_errors:
                            self.compile_errors.append(message)
                        result = self.original(model, features, mask)
            self.torch.mps.synchronize()
            return result
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.encoder_calls.append(
                {
                    "mode": self.mode,
                    "input_feature_frames": int(features.shape[1]),
                    "batch": int(features.shape[0]),
                    "bucket": bucket,
                    "fallback": fallback,
                    "latency_ms": round(elapsed, 3),
                }
            )


def run_transcription(
    speech: Any,
    experiment: EncoderExperiment,
    audio: np.ndarray,
    sample_rate: int,
    *,
    mode: str,
) -> dict[str, object]:
    from ptt_worker import normalize_text

    experiment.start_run(mode)
    started = time.perf_counter()
    result = speech.transcribe(audio=audio, sample_rate=sample_rate, timestamps="none")
    elapsed = (time.perf_counter() - started) * 1000.0
    text = normalize_text(str(result.get("text", "")))
    return {
        "mode": mode,
        "total_ms": round(elapsed, 3),
        "text": text,
        "encoder_calls": list(experiment.encoder_calls),
        "encoder_total_ms": round(
            sum(float(call["latency_ms"]) for call in experiment.encoder_calls), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path, default=Path("assets/demo-input.wav"))
    parser.add_argument("--eager-rounds", type=int, default=3)
    parser.add_argument("--hot-rounds", type=int, default=3)
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.eager_rounds < 1 or args.hot_rounds < 1:
        parser.error("round counts must be positive")
    try:
        buckets = normalize_buckets(args.buckets)
    except ValueError as exc:
        parser.error(str(exc))

    # Import torch before ptt_worker so the experiment can reject unsupported
    # versions/platforms before it downloads or loads model weights.
    import torch

    versions = check_environment(torch)
    audio, sample_rate = read_wav(args.wav)

    from kestrel.models.parakeet_tdt.model import ParakeetTdt
    from ptt_worker import load_model

    original = ParakeetTdt.encode
    experiment = EncoderExperiment(
        torch, original, buckets=buckets, fullgraph=args.fullgraph
    )

    def patched_encode(model: Any, features: Any, mask: Any) -> tuple[Any, Any]:
        return experiment.encode(model, features, mask)

    photon = None
    lock = acquire_benchmark_lock()
    try:
        ParakeetTdt.encode = patched_encode
        # Force the requested MPS device regardless of the production PTT_DEVICE.
        photon, speech = load_model("mps", announce=False)
        eager = [
            run_transcription(
                speech, experiment, audio, sample_rate, mode="eager"
            )
            for _ in range(args.eager_rounds)
        ]

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        cold = run_transcription(
            speech, experiment, audio, sample_rate, mode="compile"
        )
        hot = [
            run_transcription(
                speech, experiment, audio, sample_rate, mode="compile"
            )
            for _ in range(args.hot_rounds)
        ]
        counters = counter_snapshot(torch)
    finally:
        ParakeetTdt.encode = original
        if photon is not None:
            with contextlib.redirect_stdout(None):
                photon.__exit__(None, None, None)
        release_benchmark_lock(lock)

    baseline = eager[-1]["text"]
    equality = {
        "cold": cold["text"] == baseline,
        "hot": [run["text"] == baseline for run in hot],
        "all": cold["text"] == baseline
        and all(run["text"] == baseline for run in hot),
    }
    report = {
        "experiment": "apple-mps-fixed-shape-torch-compile-encoder",
        "versions": versions,
        "platform": platform.platform(),
        "wav": str(args.wav),
        "audio_seconds": round(len(audio) / sample_rate, 6),
        "sample_rate": sample_rate,
        "buckets": buckets,
        "compile_options": {
            "backend": "inductor",
            "dynamic": False,
            "fullgraph": args.fullgraph,
        },
        "eager_runs": eager,
        "cold_compile_run": cold,
        "hot_compile_runs": hot,
        "transcript_equality": equality,
        "dynamo_counters": counters,
        "path_fallback_counts": dict(experiment.fallbacks),
        "compile_errors": experiment.compile_errors,
        "benchmark_lock": str(LOCK_PATH),
        "lock_contention_detected": False,
        "warning": (
            "Private Kestrel 0.8 process-local experiment only; production and "
            "installed .venv sources are unchanged."
        ),
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0 if equality["all"] and not experiment.fallbacks else 2


if __name__ == "__main__":
    raise SystemExit(main())
