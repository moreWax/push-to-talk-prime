#!/usr/bin/env python3
"""Benchmark capture-path costs without loading or running an ASR model."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def timed_batches(operation: Callable[[], None], *, batches: int, batch_size: int) -> dict[str, float]:
    samples: list[float] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(batches):
            started = time.perf_counter_ns()
            for _ in range(batch_size):
                operation()
            samples.append((time.perf_counter_ns() - started) / batch_size / 1_000.0)
    finally:
        if enabled:
            gc.enable()
    return {
        "mean_us": statistics.fmean(samples),
        "p50_us": percentile(samples, 0.50),
        "p95_us": percentile(samples, 0.95),
    }


def live_buffer_trial(sample_rate: int, duration_seconds: float) -> dict[str, float | int]:
    from kestrel.models.asr.live import LiveAudioBuffer

    source = LiveAudioBuffer(sample_rate, window_seconds=180.0, update_seconds=2.0)
    callback_frames = round(sample_rate * 0.02)
    chunk = np.linspace(-0.1, 0.1, callback_frames, dtype=np.float32)
    chunk_frames = round(sample_rate * 2.0)
    left_frames = round(sample_rate * 10.0)
    right_frames = round(sample_rate * 2.0)
    total_frames = round(sample_rate * duration_seconds)
    previewed = 0
    appended = 0
    append_ns = 0
    snapshot_ns = 0
    snapshots = 0
    output_values = 0

    while appended < total_frames:
        current = chunk[: min(callback_frames, total_frames - appended)]
        started = time.perf_counter_ns()
        source.append(current)
        append_ns += time.perf_counter_ns() - started
        appended += len(current)
        while source.buffered_frames - previewed >= chunk_frames + right_frames:
            offset = max(0, previewed - left_frames)
            count = previewed + chunk_frames + right_frames - offset
            started = time.perf_counter_ns()
            decoded = source.snapshot(offset_frames=offset, frame_count=count)
            snapshot_ns += time.perf_counter_ns() - started
            output_values += int(decoded.waveform.size)
            snapshots += 1
            previewed += chunk_frames

    return {
        "append_ms": append_ns / 1_000_000.0,
        "snapshot_ms": snapshot_ns / 1_000_000.0,
        "total_ms": (append_ns + snapshot_ns) / 1_000_000.0,
        "append_calls": (total_frames + callback_frames - 1) // callback_frames,
        "snapshots": snapshots,
        "snapshot_output_values": output_values,
    }


def benchmark_live_buffer(duration_seconds: float, repeats: int) -> dict[str, Any]:
    # Discard one run per rate so imports and native resampler setup are not timed.
    live_buffer_trial(16_000, min(duration_seconds, 6.0))
    live_buffer_trial(48_000, min(duration_seconds, 6.0))
    result: dict[str, Any] = {
        "duration_seconds": duration_seconds,
        "preview_geometry_seconds": {"left": 10.0, "update": 2.0, "right": 2.0},
        "callback_chunk_ms": 20.0,
        "repeats": repeats,
    }
    for label, rate in (("pre_resampled_16k", 16_000), ("native_48k", 48_000)):
        trials = [live_buffer_trial(rate, duration_seconds) for _ in range(repeats)]
        result[label] = {
            key: (statistics.median(row[key] for row in trials) if key.endswith("_ms") else trials[0][key])
            for key in trials[0]
        }
    native = result["native_48k"]
    at_16k = result["pre_resampled_16k"]
    result["native_minus_16k_snapshot_ms"] = native["snapshot_ms"] - at_16k["snapshot_ms"]
    result["native_over_16k_total_ratio"] = native["total_ms"] / at_16k["total_ms"]
    return result


def current_rss_bytes() -> int:
    # ps reports resident size in KiB on macOS and Linux.
    value = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
    ).strip()
    return int(value) * 1024


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and most BSDs report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


class FakeStream:
    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def retention_child(duration_seconds: float) -> dict[str, Any]:
    from ptt_worker import Recorder

    recorder = Recorder(None, np)
    recorder.sample_rate = 16_000
    recorder.stream = FakeStream()
    frame_count = round(recorder.sample_rate * 0.02)
    frames = round(duration_seconds / 0.02)
    baseline = current_rss_bytes()
    started = time.perf_counter_ns()
    for index in range(frames):
        # full touches each page and prevents a shared backing allocation.
        recorder.frames.append(np.full(frame_count, (index % 17) / 100.0, dtype=np.float32))
    retention_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    retained = current_rss_bytes()
    before_stop_peak = peak_rss_bytes()
    started = time.perf_counter_ns()
    audio, sample_rate = recorder.finish()
    stop_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    post_stop = current_rss_bytes()
    peak = peak_rss_bytes()
    return {
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "callback_chunk_ms": 20.0,
        "frame_arrays": frames,
        "audio_bytes": int(audio.nbytes),
        "retention_build_ms": retention_ms,
        "concatenate_stop_ms": stop_ms,
        "retained_rss_delta_mb": (retained - baseline) / (1024 * 1024),
        "post_stop_rss_delta_mb": (post_stop - baseline) / (1024 * 1024),
        "peak_rss_delta_mb": (peak - baseline) / (1024 * 1024),
        "stop_peak_increase_mb": (peak - before_stop_peak) / (1024 * 1024),
    }


def benchmark_retention() -> list[dict[str, Any]]:
    results = []
    for seconds in (10, 60, 120):
        command = [sys.executable, str(Path(__file__).resolve()), "--retention-child", str(seconds)]
        child = subprocess.run(command, check=True, capture_output=True, text=True)
        results.append(json.loads(child.stdout))
    return results


class NullWriter:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


def benchmark_callback() -> dict[str, Any]:
    import ptt_worker

    original_output = ptt_worker._PROTOCOL_OUT
    ptt_worker._PROTOCOL_OUT = NullWriter()
    try:
        rates: dict[str, Any] = {}
        for sample_rate in (16_000, 48_000):
            mono = np.linspace(-0.2, 0.2, round(sample_rate * 0.02), dtype=np.float32)

            def rms() -> None:
                value = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
                float(np.sqrt(min(value * 16.384, 1.0)))

            payload = {"event": "level", "level": 0.73123456789}

            def json_emit() -> None:
                ptt_worker.emit(payload)

            def combined() -> None:
                value = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
                level = float(np.sqrt(min(value * 16.384, 1.0)))
                ptt_worker.emit({"event": "level", "level": level})

            rates[str(sample_rate)] = {
                "frame_values": len(mono),
                "rms_level": timed_batches(rms, batches=100, batch_size=100),
                "json_emit_to_null_writer": timed_batches(json_emit, batches=100, batch_size=100),
                "rms_plus_json_emit": timed_batches(combined, batches=100, batch_size=100),
            }
        return {"level_emit_rate_hz": 20, "rates": rates}
    finally:
        ptt_worker._PROTOCOL_OUT = original_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0, help="live-buffer input duration (default: 60)")
    parser.add_argument("--repeats", type=int, default=5, help="live-buffer trial count (default: 5)")
    parser.add_argument("--retention-child", type=float, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.retention_child is not None:
        print(json.dumps(retention_child(args.retention_child)))
        return
    if args.duration < 4:
        raise SystemExit("--duration must be at least 4 seconds")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    result = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "live_audio_buffer": benchmark_live_buffer(args.duration, args.repeats),
        "recorder_retention": benchmark_retention(),
        "callback_level": benchmark_callback(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
