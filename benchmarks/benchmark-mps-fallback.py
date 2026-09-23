#!/usr/bin/env python3
"""Probe Parakeet MPS fallbacks and host synchronization without patching Kestrel.

The process must start with PYTORCH_ENABLE_MPS_FALLBACK=0.  A successful
transcription is therefore an end-to-end unsupported-op check for that input.
Python-level tensor materializations and explicit MPS synchronizations are
wrapped temporarily to locate host waits in Kestrel's decode path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
import wave
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")
EXPECTED = {"torch": "2.14.0", "kestrel": "0.8.0", "kestrel-kernels": "0.7.0"}


@contextlib.contextmanager
def atomic_lock() -> Any:
    try:
        fd = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        owner = LOCK_PATH.read_text(errors="replace").strip()
        raise RuntimeError(f"MPS benchmark lock is held ({LOCK_PATH}: {owner})") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("input must be mono 16-bit PCM WAV")
        rate = source.getframerate()
        data = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0, rate


class HostWaitProbe:
    """Temporarily count Python-visible MPS-to-host materializations."""

    METHODS = ("tolist", "item", "cpu", "numpy", "__int__", "__float__", "__bool__")

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.phase = "startup"
        self.events: Counter[tuple[str, str, str]] = Counter()
        self.explicit_syncs: Counter[tuple[str, str]] = Counter()
        self._originals: dict[str, Callable[..., Any]] = {}
        self._original_sync: Callable[..., Any] | None = None
        self._mutex = threading.Lock()

    @staticmethod
    def _site() -> str:
        frames = traceback.extract_stack(limit=18)[:-2]
        for frame in reversed(frames):
            normalized = frame.filename.replace("\\", "/")
            if "/kestrel/" in normalized or "/kestrel_kernels/" in normalized:
                package = normalized.rsplit("site-packages/", 1)[-1]
                return f"{package}:{frame.lineno}:{frame.name}"
        if frames:
            frame = frames[-1]
            return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        return "unknown"

    def _record(self, kind: str, tensor: Any) -> None:
        if getattr(getattr(tensor, "device", None), "type", None) != "mps":
            return
        detail = f"shape={tuple(tensor.shape)},dtype={tensor.dtype}"
        with self._mutex:
            self.events[(self.phase, kind, f"{self._site()} ({detail})")] += 1

    def __enter__(self) -> "HostWaitProbe":
        tensor_type = self.torch.Tensor
        for name in self.METHODS:
            original = getattr(tensor_type, name)
            self._originals[name] = original

            def wrapped(tensor: Any, *args: Any, __name: str = name, __original: Any = original, **kwargs: Any) -> Any:
                self._record(f"Tensor.{__name}", tensor)
                return __original(tensor, *args, **kwargs)

            setattr(tensor_type, name, wrapped)
        self._original_sync = self.torch.mps.synchronize

        def synchronize(*args: Any, **kwargs: Any) -> Any:
            with self._mutex:
                self.explicit_syncs[(self.phase, self._site())] += 1
            assert self._original_sync is not None
            return self._original_sync(*args, **kwargs)

        self.torch.mps.synchronize = synchronize
        return self

    def __exit__(self, *_args: Any) -> None:
        for name, original in self._originals.items():
            setattr(self.torch.Tensor, name, original)
        if self._original_sync is not None:
            self.torch.mps.synchronize = self._original_sync

    def report(self) -> dict[str, object]:
        return {
            "host_materializations": [
                {"phase": phase, "kind": kind, "site": site, "count": count}
                for (phase, kind, site), count in sorted(self.events.items())
            ],
            "explicit_mps_synchronizations": [
                {"phase": phase, "site": site, "count": count}
                for (phase, site), count in sorted(self.explicit_syncs.items())
            ],
        }


def transcribe_streaming(speech: Any, audio: np.ndarray, rate: int, chunk_ms: int) -> dict[str, Any]:
    frames = max(1, round(rate * chunk_ms / 1000))

    async def chunks():
        for start in range(0, audio.size, frames):
            yield audio[start : start + frames]

    stream = speech.transcribe(audio=chunks(), sample_rate=rate, timestamps="none", stream=True)
    updates = 0
    for _update in stream:
        updates += 1
    result = stream.result()
    return {"text": str(result.get("text", "")), "updates": updates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path, default=Path("assets/demo-input.wav"))
    parser.add_argument("--mode", choices=("batch", "stream", "both"), default="both")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("start this process with PYTORCH_ENABLE_MPS_FALLBACK=0")

    with atomic_lock():
        import torch

        versions = {name: version(name) for name in EXPECTED}
        if versions != EXPECTED:
            raise RuntimeError(f"version mismatch: actual={versions}, expected={EXPECTED}")
        if sys.platform != "darwin" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS on macOS arm64 is required")

        audio, rate = read_wav(args.wav)
        # Keep load separate from the worker's warmup so phases are attributable.
        os.environ["PTT_SKIP_MODEL_WARMUP"] = "1"
        from ptt_worker import load_model, normalize_text

        started = time.perf_counter()
        photon = None
        runs: list[dict[str, object]] = []
        probe = HostWaitProbe(torch)
        try:
            with probe:
                probe.phase = "model_load"
                photon, speech = load_model("mps", announce=False)
                if args.mode in ("batch", "both"):
                    probe.phase = "batch_transcribe"
                    before = time.perf_counter()
                    result = speech.transcribe(audio=audio, sample_rate=rate, timestamps="none")
                    torch.mps.synchronize()
                    runs.append({
                        "mode": "batch",
                        "milliseconds": round((time.perf_counter() - before) * 1000, 3),
                        "text": normalize_text(str(result.get("text", ""))),
                    })
                if args.mode in ("stream", "both"):
                    probe.phase = "stream_transcribe"
                    before = time.perf_counter()
                    result = transcribe_streaming(speech, audio, rate, args.chunk_ms)
                    torch.mps.synchronize()
                    runs.append({
                        "mode": "stream",
                        "milliseconds": round((time.perf_counter() - before) * 1000, 3),
                        "text": normalize_text(result["text"]),
                        "updates": result["updates"],
                    })
        finally:
            if photon is not None:
                photon.__exit__(None, None, None)

        report = {
            "claim_scope": "The tested paths completed while PyTorch MPS CPU fallback was disabled; this is not proof for untested shapes or inputs.",
            "environment": {
                "PYTORCH_ENABLE_MPS_FALLBACK": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
                "platform": platform.platform(),
                "versions": versions,
                "mps_available": torch.backends.mps.is_available(),
            },
            "input": {"path": str(args.wav), "samples": int(audio.size), "sample_rate": rate},
            "runs": runs,
            "probe": probe.report(),
            "total_milliseconds": round((time.perf_counter() - started) * 1000, 3),
        }
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(rendered + "\n")
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
