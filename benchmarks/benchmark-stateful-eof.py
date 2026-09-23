#!/usr/bin/env python3
"""Benchmark experimental stateful EOF finalization against Kestrel 0.8 replay.

This script changes Kestrel functions only in this process. It does not modify the
installed package or the push-to-talk worker's production behavior.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import time
import wave
from typing import Any, Callable

import numpy as np

# Make direct `scripts/...py` execution resolve the repository worker without
# requiring installation or changing the project environment.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing the worker gives this experiment the same model, telemetry policy,
# version check, and streaming window configuration as production.
from ptt_worker import load_model, normalize_text


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("input must be mono 16-bit PCM WAV")
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


async def stateful_run_live_pcm(
    invoke: Any,
    request: Any,
    *,
    image: object | None,
    prompt: dict[str, object],
    settings: dict[str, object] | None,
    emit: Callable[[dict[str, object]], None] | None = None,
) -> Any:
    """Experimental replacement for Kestrel 0.8 longform._run_live_pcm.

    Preview decoding is unchanged in shape. At EOF, this sends only the bounded
    left context plus the uncommitted tail through the private _StreamWindow
    state path. The stock implementation instead drops preview state and sends
    the complete buffered block through the exact non-streaming path.
    """
    from kestrel.models.asr.live import LiveAudioBuffer
    from kestrel.models.parakeet_tdt import longform
    from kestrel.models.parakeet_tdt.runtime import _StreamWindow

    if request.sample_rate is None:
        raise ValueError("sample_rate is required for live PCM")
    source = LiveAudioBuffer(
        request.sample_rate,
        window_seconds=longform._STREAM_CHUNK_SECONDS,
        update_seconds=longform._LIVE_CHUNK_SECONDS,
    )
    iterator = request.audio.__aiter__()
    state = None
    last_value = None
    previewed = 0
    chunk_frames = round(float(longform._LIVE_CHUNK_SECONDS) * source.sample_rate)
    right_frames = round(float(longform._LIVE_RIGHT_SECONDS) * source.sample_rate)
    left_frames = round(float(longform._LIVE_LEFT_SECONDS) * source.sample_rate)

    async def decode(offset: int, start: int, count: int | None) -> Any:
        nonlocal state, last_value
        frame_count = source.buffered_frames - offset if count is None else start + count + right_frames - offset
        frame_count = min(frame_count, source.buffered_frames - offset)
        audio = await asyncio.to_thread(source.snapshot, offset_frames=offset, frame_count=frame_count)
        leaf = longform._leaf_prompt(prompt, audio)
        leaf["_stream_window"] = _StreamWindow(
            state,
            round((start - offset) * 16_000 / source.sample_rate),
            None if count is None else round(count * 16_000 / source.sample_rate),
            source.duration_seconds,
        )
        value = await invoke(leaf, image=image, settings=settings)
        state = value.output.pop("_stream_state", None)
        if state is None:
            raise RuntimeError("Parakeet leaf returned no streaming state")
        last_value = value
        if emit is not None:
            emit({**value.output, "provisional": count is not None})
        return value

    saw_chunk = False
    try:
        async for chunk in iterator:
            saw_chunk = True
            source.append(chunk)
            # This benchmark is intentionally limited to utterances below the
            # 180 s commit window. Avoid silently benchmarking another design.
            if source.buffered_frames >= source.window_frames + source.min_tail_frames:
                raise ValueError("stateful EOF experiment only supports one Kestrel live block")
            while source.buffered_frames - previewed >= chunk_frames + right_frames:
                offset = max(0, previewed - left_frames)
                await decode(offset, previewed, chunk_frames)
                previewed += chunk_frames
        if not saw_chunk:
            raise ValueError("live audio must yield at least one PCM chunk")
        if source.buffered_frames > previewed:
            offset = max(0, previewed - left_frames)
            await decode(offset, previewed, None)
        if last_value is None:
            raise RuntimeError("stateful EOF transcription produced no chunks")
        return last_value
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()


def run_one(
    speech: Any,
    audio: np.ndarray,
    sample_rate: int,
    *,
    mode: str,
    frame_ms: float,
    tail_ms: float,
    realtime: bool,
) -> dict[str, object]:
    import kestrel.models.parakeet_tdt.longform as longform

    frame_count = max(1, round(sample_rate * frame_ms / 1000.0))
    release_at: list[float] = []

    async def chunks():
        # Capture runs independently from inference, like Recorder's callback
        # thread and queue. Slow preview decoding must not stretch the WAV or
        # move the measured release time.
        pending: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        async def produce() -> None:
            started = time.monotonic()
            for offset in range(0, len(audio), frame_count):
                if realtime:
                    target = started + offset / sample_rate
                    await asyncio.sleep(max(0.0, target - time.monotonic()))
                await pending.put(audio[offset : offset + frame_count])
            if realtime:
                # Release occurs after the full duration of the last frame, not
                # when that frame first becomes available to the decoder.
                target = started + len(audio) / sample_rate
                await asyncio.sleep(max(0.0, target - time.monotonic()))
            release_at.append(time.monotonic())
            tail_frames = round(sample_rate * tail_ms / 1000.0)
            if tail_frames:
                await pending.put(np.zeros(tail_frames, dtype=np.float32))
            await pending.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                chunk = await pending.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    original = longform._run_live_pcm
    if mode == "stateful_eof":
        longform._run_live_pcm = stateful_run_live_pcm
    started = time.monotonic()
    try:
        stream = speech.transcribe(audio=chunks(), sample_rate=sample_rate, timestamps="none", stream=True)
        updates = sum(1 for _ in stream)
        result = stream.result()
    finally:
        longform._run_live_pcm = original
    finished = time.monotonic()
    if not release_at:
        raise RuntimeError("audio source did not reach release")
    return {
        "mode": mode,
        "text": normalize_text(str(result.get("text", ""))),
        "release_to_final_ms": round((finished - release_at[0]) * 1000.0, 2),
        "total_ms": round((finished - started) * 1000.0, 2),
        "updates": updates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path, default=Path("assets/demo-input.wav"))
    parser.add_argument("--device", default=os.getenv("PTT_DEVICE", "auto"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--tail-ms", type=float, default=320.0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    audio, sample_rate = read_wav(args.wav)
    photon, speech = load_model(args.device, announce=False)
    rows: list[dict[str, object]] = []
    try:
        for _ in range(args.rounds):
            rows.append(run_one(speech, audio, sample_rate, mode="exact_replay", frame_ms=args.frame_ms, tail_ms=args.tail_ms, realtime=not args.no_realtime))
            rows.append(run_one(speech, audio, sample_rate, mode="stateful_eof", frame_ms=args.frame_ms, tail_ms=args.tail_ms, realtime=not args.no_realtime))
    finally:
        with contextlib.redirect_stdout(None):
            photon.__exit__(None, None, None)

    pairs = [rows[index:index + 2] for index in range(0, len(rows), 2)]
    equality = [pair[0]["text"] == pair[1]["text"] for pair in pairs]
    report = {
        "kestrel": "0.8.0",
        "wav": str(args.wav),
        "audio_seconds": round(len(audio) / sample_rate, 6),
        "sample_rate": sample_rate,
        "frame_ms": args.frame_ms,
        "tail_ms": args.tail_ms,
        "realtime_feed": not args.no_realtime,
        "runs": rows,
        "exact_transcript_equality": equality,
        "all_exactly_equal": all(equality),
        "warning": "Uses private Kestrel 0.8 APIs in this process only; not production code.",
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0 if all(equality) else 2


if __name__ == "__main__":
    raise SystemExit(main())
