#!/usr/bin/env python3
"""Compare native 16 kHz capture with a 48 kHz capture/resample round trip."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import unicodedata
import wave
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")
DEFAULT_CORPUS = Path("/tmp/ptt-stream-corpus")


@contextlib.contextmanager
def exclusive_mps_lock() -> Iterator[None]:
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        owner = LOCK_PATH.read_text(errors="replace").strip()
        raise RuntimeError(f"MPS benchmark lock exists: {LOCK_PATH} ({owner})") from exc
    try:
        os.write(fd, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            LOCK_PATH.unlink()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM")
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def upsample_kaiser_3x(audio: np.ndarray, taps: int = 193, beta: float = 12.0) -> np.ndarray:
    """Band-limited 3x interpolation with a linear-phase Kaiser-windowed sinc."""
    if taps % 2 != 1:
        raise ValueError("taps must be odd")
    factor = 3
    x = np.arange(taps, dtype=np.float64) - taps // 2
    kernel = np.sinc(x / factor) * np.kaiser(taps, beta)
    kernel *= factor / kernel.sum()
    expanded = np.zeros(len(audio) * factor, dtype=np.float64)
    expanded[::factor] = audio
    # 'same' centers the odd linear-phase filter and preserves capture duration.
    return np.convolve(expanded, kernel, mode="same").astype(np.float32)


def probe_input_16k() -> dict[str, object]:
    try:
        import sounddevice as sd
        default = sd.default.device
        # sounddevice uses a private pair type rather than tuple/list.
        try:
            input_device = int(default[0])
        except (TypeError, IndexError):
            input_device = int(default)
        device = sd.query_devices(input_device, "input")
        started = time.perf_counter_ns()
        sd.check_input_settings(device=input_device, channels=1, dtype="float32", samplerate=16_000)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return {"supported": True, "device_index": input_device, "device": dict(device),
                "check_ms": round(elapsed, 3), "settings": {"channels": 1, "dtype": "float32", "samplerate": 16000}}
    except Exception as exc:
        return {"supported": False, "error_type": type(exc).__name__, "error": str(exc),
                "settings": {"channels": 1, "dtype": "float32", "samplerate": 16000}}


def validate_manifest(directory: Path) -> list[dict[str, object]]:
    records = json.loads((directory / "manifest.json").read_text())
    for record in records:
        path = directory / f"{record['id']}.wav"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        _, rate = read_wav(path)
        if rate != 16_000:
            raise ValueError(f"{path}: expected 16000 Hz, got {rate}")
    return records


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def quality(reference: str, hypothesis: str) -> dict[str, object]:
    ref, hyp = normalized_text(reference), normalized_text(hypothesis)
    return {
        "raw_exact": reference == hypothesis,
        "normalized_exact": ref == hyp,
        "normalized_word_edits": edit_distance(ref.split(), hyp.split()),
        "normalized_reference_words": len(ref.split()),
        "normalized_character_edits": edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")),
        "normalized_reference_characters": len(ref.replace(" ", "")),
    }


def run_path(speech: Any, audio: np.ndarray, sample_rate: int, *, frame_ms: float, tail_ms: float) -> dict[str, object]:
    from kestrel.models.asr.live import LiveAudioBuffer
    from kestrel.models.parakeet_tdt import longform
    from ptt_worker import InterimTranscriptStabilizer, normalize_text

    metrics: dict[str, Any] = {"append_ns": 0, "append_calls": 0, "snapshot_ns": 0,
                               "snapshot_calls": 0, "snapshot_source_frames": [],
                               "snapshot_output_frames": []}
    original_append, original_snapshot = LiveAudioBuffer.append, LiveAudioBuffer.snapshot
    original_leaf = longform._leaf_prompt
    decoded_frames: list[int] = []

    def append(self: Any, chunk: Any) -> None:
        started = time.perf_counter_ns()
        try:
            return original_append(self, chunk)
        finally:
            metrics["append_ns"] += time.perf_counter_ns() - started
            metrics["append_calls"] += 1

    def snapshot(self: Any, **kwargs: Any) -> Any:
        before = self.buffered_frames
        count = kwargs.get("frame_count")
        source_frames = min(before - kwargs.get("offset_frames", 0), self.window_frames) if count is None else count
        started = time.perf_counter_ns()
        try:
            result = original_snapshot(self, **kwargs)
            return result
        finally:
            metrics["snapshot_ns"] += time.perf_counter_ns() - started
            metrics["snapshot_calls"] += 1
            metrics["snapshot_source_frames"].append(int(source_frames))
            if "result" in locals():
                metrics["snapshot_output_frames"].append(int(result.waveform.size))

    def leaf(prompt: Any, chunk: Any) -> dict[str, object]:
        decoded_frames.append(int(chunk.waveform.shape[-1]))
        return original_leaf(prompt, chunk)

    LiveAudioBuffer.append, LiveAudioBuffer.snapshot, longform._leaf_prompt = append, snapshot, leaf
    frame_count = max(1, round(sample_rate * frame_ms / 1000))
    released_at: list[float] = []
    started_at = time.monotonic()

    async def chunks():
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        async def produce() -> None:
            capture_started = time.monotonic()
            for offset in range(0, len(audio), frame_count):
                await asyncio.sleep(max(0.0, capture_started + offset / sample_rate - time.monotonic()))
                await queue.put(audio[offset:offset + frame_count])
            await asyncio.sleep(max(0.0, capture_started + len(audio) / sample_rate - time.monotonic()))
            released_at.append(time.monotonic())
            await queue.put(np.zeros(round(sample_rate * tail_ms / 1000), dtype=np.float32))
            await queue.put(None)
        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    raw: list[str] = []
    stable: list[str] = []
    raw_times: list[float] = []
    stable_times: list[float] = []
    stabilizer = InterimTranscriptStabilizer(2)
    try:
        stream = speech.transcribe(audio=chunks(), sample_rate=sample_rate, timestamps="none", stream=True)
        for update in stream:
            if update.get("provisional") is False:
                continue
            text = normalize_text(str(update.get("text", "")))
            raw.append(text); raw_times.append(time.monotonic())
            visible = stabilizer.push(text)
            if visible is not None:
                stable.append(visible); stable_times.append(time.monotonic())
        result = stream.result()
    finally:
        LiveAudioBuffer.append, LiveAudioBuffer.snapshot, longform._leaf_prompt = original_append, original_snapshot, original_leaf
    finished = time.monotonic()
    snapshot_ms = metrics.pop("snapshot_ns") / 1e6
    append_ms = metrics.pop("append_ns") / 1e6
    source_seconds = (len(audio) + round(sample_rate * tail_ms / 1000)) / sample_rate
    decoded_seconds = sum(decoded_frames) / 16_000
    return {
        "sample_rate": sample_rate, "input_frames": len(audio), "raw_previews": raw,
        "stable_previews": stable, "final_text": normalize_text(str(result.get("text", ""))),
        "first_raw_ms": round((raw_times[0] - started_at) * 1000, 2) if raw_times else None,
        "first_stable_ms": round((stable_times[0] - started_at) * 1000, 2) if stable_times else None,
        "release_to_final_ms": round((finished - released_at[0]) * 1000, 2),
        "wall_ms": round((finished - started_at) * 1000, 2),
        "buffer_cost": {**metrics, "append_ms": round(append_ms, 4), "snapshot_ms": round(snapshot_ms, 4),
                        "snapshot_mean_ms": round(snapshot_ms / metrics["snapshot_calls"], 4) if metrics["snapshot_calls"] else None},
        "compute": {"model_invocations": len(decoded_frames), "decoded_frames_per_call": decoded_frames,
                    "decoded_input_seconds": round(decoded_seconds, 4), "source_seconds_including_tail": round(source_seconds, 4),
                    "audio_compute_amplification": round(decoded_seconds / source_seconds, 6)},
    }


def comparison(native: dict[str, object], roundtrip: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("raw_previews", "stable_previews"):
        left, right = native[name], roundtrip[name]
        assert isinstance(left, list) and isinstance(right, list)
        result[name] = {"native_count": len(left), "roundtrip_count": len(right),
                        "sequence_exact": left == right,
                        "ordinal_exact": sum(a == b for a, b in zip(left, right)),
                        "ordinal_compared": min(len(left), len(right)),
                        "first_mismatch_ordinal": next((i + 1 for i, (a, b) in enumerate(zip(left, right)) if a != b), None)}
    result["final_exact"] = native["final_text"] == roundtrip["final_text"]
    return result


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/benchmark-capture-rate-corpus-results.json"))
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--tail-ms", type=float, default=320.0)
    args = parser.parse_args()
    manifest = validate_manifest(args.corpus_dir)
    probe = probe_input_16k()
    rows: list[dict[str, object]] = []
    with exclusive_mps_lock():
        import torch
        if sys.platform != "darwin" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS on arm64 macOS is required")
        from ptt_worker import load_model
        photon = None
        try:
            photon, speech = load_model("mps", announce=False)
            for index, item in enumerate(manifest):
                audio16, rate = read_wav(args.corpus_dir / f"{item['id']}.wav")
                up_started = time.perf_counter_ns()
                audio48 = upsample_kaiser_3x(audio16)
                upsample_ms = (time.perf_counter_ns() - up_started) / 1e6
                inputs = [("native_16k", audio16, rate), ("upsampled_48k", audio48, 48_000)]
                if index % 2:
                    inputs.reverse()
                paths = {}
                for order, (name, audio, sample_rate) in enumerate(inputs, 1):
                    value = run_path(speech, audio, sample_rate, frame_ms=args.frame_ms, tail_ms=args.tail_ms)
                    value["order"] = order
                    if name == "upsampled_48k":
                        value["offline_high_quality_upsample_ms"] = round(upsample_ms, 4)
                    paths[name] = value
                for name, value in paths.items():
                    value["quality"] = quality(str(item["reference"]), str(value["final_text"]))
                rows.append({"id": item["id"], "reference": item["reference"], "audio_seconds": item["seconds"],
                             "paths": paths, "parity": comparison(paths["native_16k"], paths["upsampled_48k"])})
        finally:
            if photon is not None:
                with contextlib.redirect_stdout(None):
                    photon.__exit__(None, None, None)
    modes = ("native_16k", "upsampled_48k")
    summary: dict[str, Any] = {}
    for mode in modes:
        selected = [row["paths"][mode] for row in rows]
        summary[mode] = {
            "first_raw_median_ms": median([x["first_raw_ms"] for x in selected if x["first_raw_ms"] is not None]),
            "first_stable_median_ms": median([x["first_stable_ms"] for x in selected if x["first_stable_ms"] is not None]),
            "release_to_final_median_ms": median([x["release_to_final_ms"] for x in selected]),
            "snapshot_total_ms": round(sum(x["buffer_cost"]["snapshot_ms"] for x in selected), 4),
            "append_total_ms": round(sum(x["buffer_cost"]["append_ms"] for x in selected), 4),
            "model_invocations": sum(x["compute"]["model_invocations"] for x in selected),
            "decoded_input_seconds": round(sum(x["compute"]["decoded_input_seconds"] for x in selected), 4),
            "source_seconds_including_tail": round(sum(x["compute"]["source_seconds_including_tail"] for x in selected), 4),
            "raw_exact_matches": sum(x["quality"]["raw_exact"] for x in selected),
            "normalized_exact_matches": sum(x["quality"]["normalized_exact"] for x in selected),
        }
        summary[mode]["audio_compute_amplification"] = round(summary[mode]["decoded_input_seconds"] / summary[mode]["source_seconds_including_tail"], 6)
        word_edits = sum(x["quality"]["normalized_word_edits"] for x in selected)
        word_units = sum(x["quality"]["normalized_reference_words"] for x in selected)
        character_edits = sum(x["quality"]["normalized_character_edits"] for x in selected)
        character_units = sum(x["quality"]["normalized_reference_characters"] for x in selected)
        summary[mode]["normalized_wer"] = round(word_edits / word_units, 6)
        summary[mode]["normalized_cer"] = round(character_edits / character_units, 6)
    summary["parity"] = {
        "final_exact": sum(row["parity"]["final_exact"] for row in rows),
        "raw_preview_sequence_exact": sum(row["parity"]["raw_previews"]["sequence_exact"] for row in rows),
        "stable_preview_sequence_exact": sum(row["parity"]["stable_previews"]["sequence_exact"] for row in rows),
        "utterances": len(rows),
    }
    summary["upsample_total_ms"] = round(sum(row["paths"]["upsampled_48k"]["offline_high_quality_upsample_ms"] for row in rows), 4)
    report = {"experiment": "native-16k-vs-hq-upsample-48k-kestrel-resample", "versions": {
        "kestrel": version("kestrel"), "kestrel-kernels": version("kestrel-kernels"), "torch": version("torch")},
        "platform": platform.platform(), "corpus_dir": str(args.corpus_dir), "frame_ms": args.frame_ms,
        "tail_ms": args.tail_ms, "realtime_async_feed": True,
        "production_streaming_settings": {"chunk_ms": 160, "right_ms": 480, "left_ms": 4000, "stability": 2},
        "upsampler": {"kind": "linear-phase Kaiser-windowed sinc", "factor": 3, "taps": 193, "beta": 12.0},
        "input_settings_probe": probe, "alternating_path_order": True, "corpus": manifest, "runs": rows, "summary": summary,
        "metric_notes": ["Snapshot timing wraps Kestrel LiveAudioBuffer.snapshot and therefore includes its 48-to-16 kHz native resampling.",
                         "Amplification is total 16 kHz decoder input duration divided by source duration including the 320 ms tail.",
                         "Preview parity compares the complete ordered raw and production-stabilized transcript sequences exactly."]}
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.write_text(encoded + "\n")
    print(encoded)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
