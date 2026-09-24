#!/usr/bin/env python3
"""Benchmark one early stateless draft beside the production live stream."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import asdict, dataclass
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
import threading
import time
import unicodedata
import wave
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")
DEFAULT_CORPUS_DIR = Path("/tmp/ptt-stream-corpus")


@dataclass(frozen=True)
class Candidate:
    name: str
    preset: str
    schedule: str
    draft_ms: float | None


@dataclass(frozen=True)
class Utterance:
    id: str
    category: str
    reference: str
    say_text: str
    voice: str
    rate: int


CANDIDATES = (
    Candidate("balanced", "balanced", "none", None),
    Candidate("speculative-parallel-320", "speculative", "parallel", 320.0),
    Candidate("speculative-serialized-320", "speculative", "serialized", 320.0),
    Candidate("speculative-parallel-480", "speculative", "parallel", 480.0),
    Candidate("speculative-serialized-480", "speculative", "serialized", 480.0),
)

CORPUS = (
    Utterance("code-terms", "code_terms", "Refactor the async function and return a JSON object.",
              "Refactor the async function and return a JSON object.", "Samantha", 185),
    Utterance("filenames", "filenames", "Open README.md and edit config.yaml.",
              "Open README dot M D and edit config dot YAML.", "Daniel", 170),
    Utterance("acronyms", "acronyms", "Restart the HTTP API and verify TLS.",
              "Restart the H T T P A P I and verify T L S.", "Karen", 190),
    Utterance("numbers", "numbers", "Set port 8080 and retry in 25 seconds.",
              "Set port eight thousand eighty and retry in twenty five seconds.", "Moira", 180),
    Utterance("punctuation", "punctuation", "Print hello, world, then exit.",
              "Print hello, world, then exit.", "Tessa", 175),
    Utterance("short-command", "short_command", "Run tests.",
              "Run tests.", "Samantha", 210),
    Utterance("long-command", "long_command",
              "Search the repository for deprecated flags, update every call site, run the unit tests, and summarize the failures.",
              "Search the repository for deprecated flags, update every call site, run the unit tests, and summarize the failures.",
              "Daniel", 165),
    Utterance("git-command", "code_terms", "Create a feature branch, commit the parser fix, and push it to origin.",
              "Create a feature branch, commit the parser fix, and push it to origin.", "Karen", 200),
    Utterance("cli-terms", "code_terms", "Run pytest with the verbose flag, then inspect stderr.",
              "Run pie test with the verbose flag, then inspect standard error.", "Moira", 155),
    Utterance("mixed-file", "filenames", "Rename server_test.go to server_integration_test.go.",
              "Rename server underscore test dot go to server underscore integration underscore test dot go.",
              "Tessa", 180),
    Utterance("paused-command", "pause", "Build the release binary, then wait for the checksum.",
              "Build the release binary, [[slnc 700]] then wait for the checksum.", "Samantha", 185),
    Utterance("database-command", "long_command",
              "Back up the database before applying migration 42, and do not restart the primary server.",
              "Back up the database before applying migration forty two, and do not restart the primary server.",
              "Daniel", 175),
)


@contextlib.contextmanager
def exclusive_mps_lock() -> Iterator[None]:
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


def reuse_corpus(directory: Path) -> list[dict[str, object]]:
    manifest_path = directory / "manifest.json"
    records = json.loads(manifest_path.read_text())
    expected_ids = [item.id for item in CORPUS]
    actual_ids = [record.get("id") for record in records]
    if actual_ids != expected_ids:
        raise RuntimeError(f"unexpected corpus IDs in {manifest_path}: {actual_ids}")
    for item, record in zip(CORPUS, records):
        expected_recipe = asdict(item)
        actual_recipe = {key: record.get(key) for key in expected_recipe}
        if actual_recipe != expected_recipe:
            raise RuntimeError(f"corpus recipe mismatch for {item.id}")
        path = directory / f"{record['id']}.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            raise RuntimeError(f"corpus hash mismatch: {path}")
    return records


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character if character.isalnum() else " " for character in value)
    return " ".join(value.split())


def error_counts(reference: str, hypothesis: str) -> dict[str, int | bool]:
    ref_words = normalized_text(reference).split()
    hyp_words = normalized_text(hypothesis).split()
    return {
        "normalized_match": ref_words == hyp_words,
        "normalized_word_edits": edit_distance(ref_words, hyp_words),
        "normalized_reference_words": len(ref_words),
    }


def prefix_counts(draft: str, final_text: str) -> dict[str, int | bool]:
    draft_words = normalized_text(draft).split()
    final_words = normalized_text(final_text).split()
    common = 0
    for draft_word, final_word in zip(draft_words, final_words):
        if draft_word != final_word:
            break
        common += 1
    return {
        "is_final_word_prefix": common == len(draft_words),
        "draft_words": len(draft_words),
        "common_prefix_words": common,
        "wrong_or_extra_words": len(draft_words) - common,
        "word_edits_to_final": edit_distance(draft_words, final_words),
    }


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
        for left_character, right_character in zip(left, right):
            if left_character != right_character:
                break
            common += 1
        retracted += len(left) - common
    return {
        "snapshot_count": len(snapshots),
        "changed_transitions": changed,
        "character_edits": edits,
        "retracted_characters": retracted,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def run_candidate(
    speech: Any,
    audio: np.ndarray,
    sample_rate: int,
    candidate: Candidate,
    *,
    frame_ms: float,
    tail_ms: float,
) -> dict[str, object]:
    from kestrel.models.parakeet_tdt import longform
    from ptt_worker import InterimTranscriptStabilizer, _IntegralFrameSeconds, normalize_text

    # These are the authoritative production settings for every candidate.
    longform._LIVE_CHUNK_SECONDS = _IntegralFrameSeconds(0.160)
    longform._LIVE_RIGHT_SECONDS = _IntegralFrameSeconds(0.480)
    longform._LIVE_LEFT_SECONDS = _IntegralFrameSeconds(4.000)

    frame_count = max(1, round(sample_rate * frame_ms / 1000.0))
    tail_frames = round(sample_rate * tail_ms / 1000.0)
    call_context = threading.local()
    original_leaf_prompt = longform._leaf_prompt
    authoritative_decoded_frames: list[int] = []
    preview_state: dict[str, object] = {
        "launched_at": None,
        "finished_at": None,
        "text": "",
        "error": None,
    }
    preview_thread: list[threading.Thread] = []
    preview_invocations: list[float] = []
    capture_started: list[float] = []
    started = time.monotonic()

    def measured_leaf_prompt(prompt: Any, chunk: Any) -> dict[str, object]:
        if not getattr(call_context, "preview", False):
            authoritative_decoded_frames.append(int(chunk.waveform.shape[-1]))
        return original_leaf_prompt(prompt, chunk)

    def preview() -> None:
        draft_frames = round(sample_rate * float(candidate.draft_ms) / 1000.0)
        preview_state["launched_at"] = time.monotonic()
        preview_invocations.append(float(preview_state["launched_at"]))
        call_context.preview = True
        try:
            result = speech.transcribe(
                audio=audio[:draft_frames].copy(),
                sample_rate=sample_rate,
                timestamps="none",
                stream=False,
            )
            preview_state["text"] = normalize_text(str(result.get("text", "")))
        except BaseException as exc:  # Preserve failures in the report before raising.
            preview_state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            preview_state["finished_at"] = time.monotonic()

    released_at: list[float] = []
    queue_depths: list[int] = []
    consume_lags_ms: list[float] = []

    async def chunks():
        pending: asyncio.Queue[tuple[np.ndarray | None, float]] = asyncio.Queue()

        async def produce() -> None:
            capture_started.append(time.monotonic())
            capture_origin = capture_started[0]
            draft_frames = (
                round(sample_rate * float(candidate.draft_ms) / 1000.0)
                if candidate.draft_ms is not None else None
            )
            launched = False
            for offset in range(0, len(audio), frame_count):
                item = audio[offset : offset + frame_count]
                # A frame becomes available at its end, not its beginning.
                available_at = capture_origin + (offset + len(item)) / sample_rate
                await asyncio.sleep(max(0.0, available_at - time.monotonic()))
                await pending.put((item, available_at))
                queue_depths.append(pending.qsize())
                captured = offset + len(item)
                if (
                    candidate.schedule == "parallel"
                    and draft_frames is not None
                    and not launched
                    and captured >= draft_frames
                ):
                    launched = True
                    thread = threading.Thread(target=preview, name=f"draft-{candidate.name}")
                    preview_thread.append(thread)
                    thread.start()
            released_at.append(time.monotonic())
            if tail_frames:
                await pending.put((np.zeros(tail_frames, dtype=np.float32), time.monotonic()))
                queue_depths.append(pending.qsize())
            await pending.put((None, time.monotonic()))

        producer = asyncio.create_task(produce())
        consumed_frames = 0
        serialized_preview_started = False
        draft_frames = (
            round(sample_rate * float(candidate.draft_ms) / 1000.0)
            if candidate.draft_ms is not None else None
        )
        try:
            while True:
                item, available_at = await pending.get()
                if item is None:
                    break
                next_consumed_frames = consumed_frames + len(item)
                if (
                    candidate.schedule == "serialized"
                    and draft_frames is not None
                    and not serialized_preview_started
                    and next_consumed_frames >= draft_frames
                ):
                    serialized_preview_started = True
                    # The live engine is awaiting source audio here, so no live
                    # invoke overlaps this stateless call. Capture continues.
                    await asyncio.to_thread(preview)
                consumed_frames = next_consumed_frames
                consume_lags_ms.append(max(0.0, (time.monotonic() - available_at) * 1000.0))
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    stabilizer = InterimTranscriptStabilizer(2)
    raw_events: list[tuple[float, str]] = []
    stable_events: list[tuple[float, str]] = []
    longform._leaf_prompt = measured_leaf_prompt
    try:
        stream = speech.transcribe(
            audio=chunks(), sample_rate=sample_rate, timestamps="none", stream=True
        )
        for update in stream:
            if update.get("provisional") is False:
                continue
            now = time.monotonic()
            text = normalize_text(str(update.get("text", "")))
            raw_events.append((now, text))
            visible = stabilizer.push(text)
            if visible is not None:
                stable_events.append((now, visible))
        result = stream.result()
        authoritative_finished = time.monotonic()
        for thread in preview_thread:
            thread.join()
    finally:
        longform._leaf_prompt = original_leaf_prompt

    if not released_at:
        raise RuntimeError("audio producer never reached release")
    expected_previews = 0 if candidate.draft_ms is None else 1
    if len(preview_invocations) != expected_previews:
        raise RuntimeError(
            f"expected {expected_previews} preview calls, launched {len(preview_invocations)}"
        )
    if preview_state["error"] is not None:
        raise RuntimeError(f"preview failed: {preview_state['error']}")

    if not capture_started:
        raise RuntimeError("capture clock never started")
    capture_origin = capture_started[0]
    final_text = normalize_text(str(result.get("text", "")))
    first_raw = next((when for when, text in raw_events if text), None)
    first_stable = stable_events[0][0] if stable_events else None
    draft_text = str(preview_state["text"])
    draft_finished = preview_state["finished_at"]
    visible_events = list(stable_events)
    if draft_finished is not None and draft_text:
        visible_events.append((float(draft_finished), draft_text))
    visible_events.sort(key=lambda event: event[0])
    decoded_seconds = sum(authoritative_decoded_frames) / 16_000.0
    source_seconds = (len(audio) + tail_frames) / sample_rate

    return {
        "candidate": asdict(candidate),
        "authoritative": {
            "first_raw_ms": round((first_raw - capture_origin) * 1000.0, 2) if first_raw else None,
            "first_stable_ms": round((first_stable - capture_origin) * 1000.0, 2) if first_stable else None,
            "release_to_final_ms": round((authoritative_finished - released_at[0]) * 1000.0, 2),
            "wall_to_final_ms": round((authoritative_finished - capture_origin) * 1000.0, 2),
            "final_text": final_text,
            "raw_churn": churn([text for _, text in raw_events]),
            "stable_churn": churn([text for _, text in stable_events]),
            "compute": {
                "model_invocations": len(authoritative_decoded_frames),
                "decoded_input_seconds": round(decoded_seconds, 4),
                "source_seconds_including_tail": round(source_seconds, 4),
                "audio_compute_amplification": round(decoded_seconds / source_seconds, 6),
            },
        },
        "preview": {
            "launched": candidate.draft_ms is not None,
            "stateless_invocations": len(preview_invocations),
            "launch_ms": (
                round((float(preview_state["launched_at"]) - capture_origin) * 1000.0, 2)
                if preview_state["launched_at"] is not None else None
            ),
            "visible_ms": (
                round((float(draft_finished) - capture_origin) * 1000.0, 2)
                if draft_finished is not None and draft_text else None
            ),
            "compute_wall_ms": (
                round((float(draft_finished) - float(preview_state["launched_at"])) * 1000.0, 2)
                if draft_finished is not None and preview_state["launched_at"] is not None else None
            ),
            "input_seconds": round(float(candidate.draft_ms) / 1000.0, 3) if candidate.draft_ms else 0.0,
            "text": draft_text,
            "nonempty": bool(draft_text),
            "visible_before_first_raw": bool(draft_text and first_raw and float(draft_finished) < first_raw),
            "visible_before_first_stable": bool(draft_text and first_stable and float(draft_finished) < first_stable),
            "errors_vs_reference": None,
            "prefix_vs_final": prefix_counts(draft_text, final_text) if draft_text else None,
        },
        "first_visible_ms": (
            round((visible_events[0][0] - capture_origin) * 1000.0, 2)
            if visible_events else None
        ),
        "combined_visible_churn": churn([text for _, text in visible_events]),
        "backlog": {
            "max_queued_chunks": max(queue_depths, default=0),
            "max_queued_audio_ms": round(max(queue_depths, default=0) * frame_ms, 3),
            "max_consume_lag_ms": round(max(consume_lags_ms, default=0.0), 3),
            "mean_consume_lag_ms": round(statistics.fmean(consume_lags_ms), 3) if consume_lags_ms else 0.0,
        },
    }


def metric(values: list[float]) -> dict[str, float | int | None]:
    return {
        "available": len(values),
        "median": round(statistics.median(values), 3) if values else None,
        "p95": percentile(values, 0.95),
        "mean": round(statistics.fmean(values), 3) if values else None,
    }


def aggregate(rows: list[dict[str, object]], candidate: Candidate) -> dict[str, object]:
    selected = [row for row in rows if row["candidate"]["name"] == candidate.name]  # type: ignore[index]
    def authoritative_values(key: str) -> list[float]:
        return [float(row["authoritative"][key]) for row in selected if row["authoritative"][key] is not None]  # type: ignore[index]
    previews = [row["preview"] for row in selected]  # type: ignore[index]
    launched = [preview for preview in previews if preview["launched"]]
    nonempty = [preview for preview in launched if preview["nonempty"]]
    preview_words = sum(int(preview["errors_vs_reference"]["normalized_reference_words"]) for preview in nonempty)  # type: ignore[index]
    preview_edits = sum(int(preview["errors_vs_reference"]["normalized_word_edits"]) for preview in nonempty)  # type: ignore[index]
    return {
        "utterances": len(selected),
        "authoritative_first_raw_ms": metric(authoritative_values("first_raw_ms")),
        "authoritative_first_stable_ms": metric(authoritative_values("first_stable_ms")),
        "authoritative_release_to_final_ms": metric(authoritative_values("release_to_final_ms")),
        "authoritative_wall_to_final_ms": metric(authoritative_values("wall_to_final_ms")),
        "first_visible_ms": metric([
            float(row["first_visible_ms"]) for row in selected if row["first_visible_ms"] is not None
        ]),
        "backlog_max_queued_audio_ms": metric([float(row["backlog"]["max_queued_audio_ms"]) for row in selected]),  # type: ignore[index]
        "backlog_max_consume_lag_ms": metric([float(row["backlog"]["max_consume_lag_ms"]) for row in selected]),  # type: ignore[index]
        "preview": {
            "launched": len(launched),
            "nonempty": len(nonempty),
            "nonempty_coverage": round(len(nonempty) / len(launched), 6) if launched else None,
            "visible_ms": metric([float(preview["visible_ms"]) for preview in nonempty]),
            "compute_wall_ms": metric([float(preview["compute_wall_ms"]) for preview in launched]),
            "input_seconds_total": round(sum(float(preview["input_seconds"]) for preview in launched), 3),
            "visible_before_first_raw": sum(bool(preview["visible_before_first_raw"]) for preview in nonempty),
            "visible_before_first_stable": sum(bool(preview["visible_before_first_stable"]) for preview in nonempty),
            "normalized_wer_vs_reference": round(preview_edits / preview_words, 6) if preview_words else None,
            "final_word_prefix": sum(bool(preview["prefix_vs_final"]["is_final_word_prefix"]) for preview in nonempty),  # type: ignore[index]
            "prefix_common_words": sum(int(preview["prefix_vs_final"]["common_prefix_words"]) for preview in nonempty),  # type: ignore[index]
            "prefix_draft_words": sum(int(preview["prefix_vs_final"]["draft_words"]) for preview in nonempty),  # type: ignore[index]
        },
        "combined_visible_churn": {
            key: sum(int(row["combined_visible_churn"][key]) for row in selected)  # type: ignore[index]
            for key in ("snapshot_count", "changed_transitions", "character_edits", "retracted_characters")
        },
        "authoritative_compute": {
            "model_invocations": sum(int(row["authoritative"]["compute"]["model_invocations"]) for row in selected),  # type: ignore[index]
            "decoded_input_seconds": round(sum(float(row["authoritative"]["compute"]["decoded_input_seconds"]) for row in selected), 4),  # type: ignore[index]
        },
        "total_logical_compute": {
            "stateless_invocations": sum(int(preview["stateless_invocations"]) for preview in previews),
            "decoded_input_seconds": round(
                sum(float(row["authoritative"]["compute"]["decoded_input_seconds"]) for row in selected)  # type: ignore[index]
                + sum(float(preview["input_seconds"]) for preview in previews),
                4,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--tail-ms", type=float, default=320.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frame_ms != 20.0:
        parser.error("this experiment requires the specified 20 ms real-time feed")
    if args.tail_ms < 0:
        parser.error("--tail-ms must be non-negative")
    manifest = reuse_corpus(args.corpus_dir)

    rows: list[dict[str, object]] = []
    with exclusive_mps_lock():
        import torch
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("this benchmark requires macOS on arm64")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available")
        from kestrel.models.parakeet_tdt import longform
        from ptt_worker import load_model
        original_windows = (
            longform._LIVE_CHUNK_SECONDS,
            longform._LIVE_RIGHT_SECONDS,
            longform._LIVE_LEFT_SECONDS,
        )
        photon = None
        stream_environment = {
            "PTT_STREAM_CHUNK_MS": "160",
            "PTT_STREAM_RIGHT_MS": "480",
            "PTT_STREAM_LEFT_MS": "4000",
        }
        saved_environment = {key: os.environ.get(key) for key in (*stream_environment, "PTT_SKIP_MODEL_WARMUP")}
        try:
            os.environ.update(stream_environment)
            os.environ.pop("PTT_SKIP_MODEL_WARMUP", None)
            photon, speech = load_model("mps", announce=False)
            for key, value in saved_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            for item_index, item in enumerate(CORPUS):
                audio, sample_rate = read_wav(args.corpus_dir / f"{item.id}.wav")
                rotation = item_index % len(CANDIDATES)
                order = CANDIDATES[rotation:] + CANDIDATES[:rotation]
                for order_index, candidate in enumerate(order):
                    print(f"[{item_index + 1:02d}/12 {order_index + 1}/5] {item.id} {candidate.name}", file=sys.stderr, flush=True)
                    row = run_candidate(
                        speech, audio, sample_rate, candidate,
                        frame_ms=args.frame_ms, tail_ms=args.tail_ms,
                    )
                    row.update({
                        "utterance": asdict(item),
                        "audio_seconds": round(len(audio) / sample_rate, 6),
                        "order_in_utterance": order_index + 1,
                    })
                    row["authoritative"]["errors_vs_reference"] = error_counts(item.reference, str(row["authoritative"]["final_text"]))  # type: ignore[index]
                    if row["preview"]["nonempty"]:  # type: ignore[index]
                        row["preview"]["errors_vs_reference"] = error_counts(item.reference, str(row["preview"]["text"]))  # type: ignore[index]
                    rows.append(row)
        finally:
            for key, value in saved_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            (
                longform._LIVE_CHUNK_SECONDS,
                longform._LIVE_RIGHT_SECONDS,
                longform._LIVE_LEFT_SECONDS,
            ) = original_windows
            if photon is not None:
                with contextlib.redirect_stdout(None):
                    photon.__exit__(None, None, None)

    balanced_final = {
        str(row["utterance"]["id"]): str(row["authoritative"]["final_text"])  # type: ignore[index]
        for row in rows if row["candidate"]["name"] == "balanced"  # type: ignore[index]
    }
    for row in rows:
        baseline = balanced_final[str(row["utterance"]["id"])]  # type: ignore[index]
        final_text = str(row["authoritative"]["final_text"])  # type: ignore[index]
        row["authoritative"]["exact_final_equals_balanced"] = final_text == baseline  # type: ignore[index]
        row["authoritative"]["normalized_final_equals_balanced"] = normalized_text(final_text) == normalized_text(baseline)  # type: ignore[index]

    aggregates = {candidate.name: aggregate(rows, candidate) for candidate in CANDIDATES}
    balanced = aggregates["balanced"]
    for candidate in CANDIDATES:
        aggregate_row = aggregates[candidate.name]
        selected = [row for row in rows if row["candidate"]["name"] == candidate.name]  # type: ignore[index]
        aggregate_row["final_parity_with_balanced"] = {
            "exact": sum(bool(row["authoritative"]["exact_final_equals_balanced"]) for row in selected),  # type: ignore[index]
            "normalized": sum(bool(row["authoritative"]["normalized_final_equals_balanced"]) for row in selected),  # type: ignore[index]
            "utterances": len(selected),
        }
        if candidate.name != "balanced":
            impacts: dict[str, object] = {}
            for key in (
                "authoritative_first_raw_ms",
                "authoritative_first_stable_ms",
                "authoritative_release_to_final_ms",
                "backlog_max_consume_lag_ms",
            ):
                candidate_p95 = aggregate_row[key]["p95"]  # type: ignore[index]
                balanced_p95 = balanced[key]["p95"]  # type: ignore[index]
                delta = float(candidate_p95) - float(balanced_p95)
                threshold = max(100.0, 0.10 * float(balanced_p95))
                impacts[key] = {
                    "p95_delta_ms": round(delta, 3),
                    "material_threshold_ms": round(threshold, 3),
                    "material_regression": delta > threshold,
                }
            aggregate_row["impact_vs_balanced"] = impacts
            aggregate_row["added_logical_compute_vs_balanced_seconds"] = round(
                float(aggregate_row["total_logical_compute"]["decoded_input_seconds"])  # type: ignore[index]
                - float(balanced["total_logical_compute"]["decoded_input_seconds"]),  # type: ignore[index]
                4,
            )
            aggregate_row["reject_for_authoritative_p95"] = any(
                bool(value["material_regression"]) for value in impacts.values()  # type: ignore[union-attr]
            )

    report = {
        "experiment": "single-model-early-draft-preview-apple-mps",
        "versions": {
            "kestrel": version("kestrel"),
            "kestrel-kernels": version("kestrel-kernels"),
            "torch": version("torch"),
        },
        "platform": platform.platform(),
        "macos_product_version": subprocess.run(
            ["sw_vers", "-productVersion"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "corpus_dir": str(args.corpus_dir),
        "corpus_reused": True,
        "corpus": manifest,
        "frame_ms": args.frame_ms,
        "tail_ms": args.tail_ms,
        "realtime_feed": True,
        "authoritative_settings": {"chunk_ms": 160, "right_ms": 480, "left_ms": 4000, "stability": 2},
        "single_warm_photon_engine": True,
        "mps_lock": str(LOCK_PATH),
        "rotating_candidate_order": True,
        "order_schedule": "balanced/speculative variants rotate one position per utterance across all five candidates",
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "runs": rows,
        "aggregates": aggregates,
        "rejection_rule": "Reject a speculative variant when any authoritative p95 latency or backlog lag rises by more than max(100 ms, 10%) versus balanced.",
        "metric_notes": [
            "Each speculative run launches exactly one stateless timestamps-none call after 320 ms or 480 ms of audio has been captured.",
            "Parallel launches the stateless call from the capture producer and permits overlap with live inference. Serialized pauses before yielding the cutoff-crossing live chunk, runs the stateless call while capture continues, and resumes live inference without model-call overlap.",
            "The 20 ms producer schedules each frame at the end of its capture interval. Queue depth and consume lag measure live-stream backlog.",
            "Draft visible time is stateless-call completion from capture start and only counts non-empty text.",
            "Prefix correctness compares normalized whole words with that run's authoritative final. WER uses the fixed corpus reference.",
            "Combined churn orders the draft completion and authoritative stable events by actual wall time.",
            "Added draft input compute is 0.32 or 0.48 decoded audio seconds per launch; compute_wall_ms measures its occupied wall interval.",
            "Final parity is exact and normalized against the matched balanced run for the same utterance.",
        ],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
