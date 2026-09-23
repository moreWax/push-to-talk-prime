#!/usr/bin/env python3
"""Benchmark process-local Kestrel streaming presets on Apple MPS.

The benchmark changes private Kestrel window constants only in this process. It
does not modify the worker, its environment, or production defaults.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import wave
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")


@dataclass(frozen=True)
class Preset:
    name: str
    chunk_ms: float
    right_ms: float
    left_ms: float
    stability: int


# This is deliberately small. It spans cadence and stabilization without a
# long combinatorial MPS run.
DEFAULT_PRESETS = (
    Preset("responsive", 80.0, 320.0, 2000.0, 1),
    Preset("production", 160.0, 480.0, 4000.0, 2),
    Preset("conservative", 320.0, 640.0, 4000.0, 3),
)

# Focused follow-up around the production setting. Names sort by the dimensions
# they encode, which also makes result inspection less error-prone.
FOCUSED_PRESETS = tuple(
    Preset(f"c160-r{right}-l{left}-s2", 160.0, float(right), float(left), 2)
    for left in (1000, 2000, 4000)
    for right in (320, 480)
) + (Preset("c240-r480-l2000-s2", 240.0, 480.0, 2000.0, 2),)

EXPECTED_TRANSCRIPT = (
    "Refactor the authentication middleware and add tests for expired tokens, "
    "malformed tokens, and missing tokens."
)


@contextlib.contextmanager
def exclusive_mps_lock() -> Iterator[None]:
    """Create an atomic, process-owned lock and remove it on normal exit."""
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        owner = LOCK_PATH.read_text(errors="replace").strip()
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
            raise ValueError("input must be mono 16-bit PCM WAV")
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 2)


def cadence(times: list[float]) -> dict[str, object]:
    gaps = [(right - left) * 1000.0 for left, right in zip(times, times[1:])]
    return {
        "count": len(times),
        "gap_count": len(gaps),
        "mean_gap_ms": round(statistics.fmean(gaps), 2) if gaps else None,
        "p50_gap_ms": percentile(gaps, 0.50),
        "p95_gap_ms": percentile(gaps, 0.95),
        "min_gap_ms": round(min(gaps), 2) if gaps else None,
        "max_gap_ms": round(max(gaps), 2) if gaps else None,
    }


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def churn(snapshots: list[str]) -> dict[str, int]:
    edits = 0
    retracted = 0
    changed = 0
    for left, right in zip(snapshots, snapshots[1:]):
        if left == right:
            continue
        changed += 1
        edits += edit_distance(left, right)
        common = 0
        for a, b in zip(left, right):
            if a != b:
                break
            common += 1
        retracted += len(left) - common
    return {
        "snapshot_count": len(snapshots),
        "changed_transitions": changed,
        "character_edits": edits,
        "retracted_characters": retracted,
    }


def run_preset(
    speech: Any,
    audio: np.ndarray,
    sample_rate: int,
    preset: Preset,
    *,
    frame_ms: float,
    tail_ms: float,
) -> dict[str, object]:
    from kestrel.models.parakeet_tdt import longform
    from ptt_worker import InterimTranscriptStabilizer, _IntegralFrameSeconds, normalize_text

    longform._LIVE_CHUNK_SECONDS = _IntegralFrameSeconds(preset.chunk_ms / 1000.0)
    longform._LIVE_RIGHT_SECONDS = _IntegralFrameSeconds(preset.right_ms / 1000.0)
    longform._LIVE_LEFT_SECONDS = _IntegralFrameSeconds(preset.left_ms / 1000.0)

    decoded_frames: list[int] = []
    original_leaf_prompt = longform._leaf_prompt

    def measured_leaf_prompt(prompt: Any, chunk: Any) -> dict[str, object]:
        decoded_frames.append(int(chunk.waveform.shape[-1]))
        return original_leaf_prompt(prompt, chunk)

    frame_count = max(1, round(sample_rate * frame_ms / 1000.0))
    release_at: list[float] = []
    started = time.monotonic()

    async def chunks():
        pending: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        async def produce() -> None:
            capture_started = time.monotonic()
            for offset in range(0, len(audio), frame_count):
                target = capture_started + offset / sample_rate
                await asyncio.sleep(max(0.0, target - time.monotonic()))
                await pending.put(audio[offset : offset + frame_count])
            target = capture_started + len(audio) / sample_rate
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            release_at.append(time.monotonic())
            tail_frames = round(sample_rate * tail_ms / 1000.0)
            if tail_frames:
                await pending.put(np.zeros(tail_frames, dtype=np.float32))
            await pending.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await pending.get()
                if item is None:
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    stabilizer = InterimTranscriptStabilizer(preset.stability)
    update_times: list[float] = []
    stable_times: list[float] = []
    raw_snapshots: list[str] = []
    stable_snapshots: list[str] = []
    longform._leaf_prompt = measured_leaf_prompt
    try:
        stream = speech.transcribe(
            audio=chunks(), sample_rate=sample_rate, timestamps="none", stream=True
        )
        for update in stream:
            now = time.monotonic()
            if update.get("provisional") is False:
                continue
            update_times.append(now)
            text = normalize_text(str(update.get("text", "")))
            raw_snapshots.append(text)
            visible = stabilizer.push(text)
            if visible is not None:
                stable_times.append(now)
                stable_snapshots.append(visible)
        result = stream.result()
    finally:
        longform._leaf_prompt = original_leaf_prompt
    finished = time.monotonic()
    if not release_at:
        raise RuntimeError("audio producer never reached release")

    final_text = normalize_text(str(result.get("text", "")))
    source_seconds = (len(audio) + round(sample_rate * tail_ms / 1000.0)) / sample_rate
    decoded_seconds = sum(decoded_frames) / 16_000.0
    first_raw = next((when for when, text in zip(update_times, raw_snapshots) if text), None)
    return {
        "preset": asdict(preset),
        "first_raw_interim_ms": round((first_raw - started) * 1000.0, 2) if first_raw else None,
        "first_stable_interim_ms": round((stable_times[0] - started) * 1000.0, 2) if stable_times else None,
        "raw_update_cadence": cadence(update_times),
        "stable_update_cadence": cadence(stable_times),
        "release_to_final_ms": round((finished - release_at[0]) * 1000.0, 2),
        "wall_ms": round((finished - started) * 1000.0, 2),
        "final_text": final_text,
        "last_visible_interim": stable_snapshots[-1] if stable_snapshots else "",
        "raw_churn": churn(raw_snapshots),
        "stable_churn": churn(stable_snapshots),
        "compute": {
            "model_invocations": len(decoded_frames),
            "decoded_input_seconds": round(decoded_seconds, 4),
            "source_seconds_including_tail": round(source_seconds, 4),
            "audio_compute_amplification": round(decoded_seconds / source_seconds, 3),
            "decoded_frames_per_call": decoded_frames,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path, default=Path("assets/demo-input.wav"))
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--tail-ms", type=float, default=320.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matrix", choices=("initial", "focused"), default="initial")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--skip-optional-240", action="store_true",
        help="omit the optional 240/480/2000 focused point",
    )
    args = parser.parse_args()
    if args.frame_ms <= 0 or args.tail_ms < 0:
        parser.error("--frame-ms must be positive and --tail-ms must be non-negative")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    presets = DEFAULT_PRESETS
    if args.matrix == "focused":
        presets = FOCUSED_PRESETS[:-1] if args.skip_optional_240 else FOCUSED_PRESETS

    with exclusive_mps_lock():
        import torch
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("this benchmark requires macOS on arm64")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available")

        audio, sample_rate = read_wav(args.wav)
        # load_model applies production defaults first. All later mutations are
        # private, process-local benchmark settings and are restored below.
        from kestrel.models.parakeet_tdt import longform
        from ptt_worker import load_model

        original_windows = (
            longform._LIVE_CHUNK_SECONDS,
            longform._LIVE_RIGHT_SECONDS,
            longform._LIVE_LEFT_SECONDS,
        )
        photon = None
        try:
            photon, speech = load_model("mps", announce=False)
            rows = []
            for round_index in range(args.rounds):
                round_presets = presets if round_index % 2 == 0 else tuple(reversed(presets))
                for order_index, preset in enumerate(round_presets):
                    row = run_preset(
                        speech, audio, sample_rate, preset,
                        frame_ms=args.frame_ms, tail_ms=args.tail_ms,
                    )
                    row["round"] = round_index + 1
                    row["order_in_round"] = order_index + 1
                    rows.append(row)
        finally:
            (
                longform._LIVE_CHUNK_SECONDS,
                longform._LIVE_RIGHT_SECONDS,
                longform._LIVE_LEFT_SECONDS,
            ) = original_windows
            if photon is not None:
                with contextlib.redirect_stdout(None):
                    photon.__exit__(None, None, None)

    baseline = rows[0]["final_text"]
    for row in rows:
        row["final_equals_baseline"] = row["final_text"] == baseline
        row["final_equals_expected"] = row["final_text"] == EXPECTED_TRANSCRIPT
        # Retain the original result key for comparison with earlier grid data.
        row["final_equals_production"] = row["final_equals_expected"]
    report = {
        "experiment": f"current-streaming-context-grid-{args.matrix}-apple-mps",
        "versions": {
            "kestrel": version("kestrel"),
            "kestrel-kernels": version("kestrel-kernels"),
            "torch": version("torch"),
        },
        "platform": platform.platform(),
        "wav": str(args.wav),
        "audio_seconds": round(len(audio) / sample_rate, 6),
        "sample_rate": sample_rate,
        "frame_ms": args.frame_ms,
        "tail_ms": args.tail_ms,
        "realtime_feed": True,
        "matrix": args.matrix,
        "rounds": args.rounds,
        "alternating_order": args.rounds > 1,
        "expected_transcript": EXPECTED_TRANSCRIPT,
        "runs": rows,
        "all_final_transcripts_equal": all(row["final_equals_baseline"] for row in rows),
        "all_final_transcripts_expected": all(row["final_equals_expected"] for row in rows),
        "notes": [
            "Compute amplification is summed model-input audio / source audio including synthetic tail.",
            "One warm model; focused rounds alternate forward and reverse order.",
            "Private Kestrel constants are changed only in this process and restored on exit.",
        ],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0 if (
        report["all_final_transcripts_equal"] and report["all_final_transcripts_expected"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
