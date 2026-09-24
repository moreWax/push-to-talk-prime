#!/usr/bin/env python3
"""Compare first-visible latency and quality across production streaming presets."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
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
DEFAULT_CORPUS_DIR = Path("/tmp/ptt-stream-corpus")


@dataclass(frozen=True)
class Preset:
    name: str
    chunk_ms: float
    right_ms: float
    left_ms: float
    stability: int


@dataclass(frozen=True)
class Utterance:
    id: str
    category: str
    reference: str
    say_text: str
    voice: str
    rate: int


PRESETS = (
    Preset("A-baseline-160-480-4000-s2", 160.0, 480.0, 4000.0, 2),
    Preset("B-160-320-4000-s2", 160.0, 320.0, 4000.0, 2),
    Preset("C-160-480-4000-s1", 160.0, 480.0, 4000.0, 1),
)

# The spoken input is fixed. Voices and rates deliberately cover five English
# locales and both short and long utterances. The pause markup inserts silence
# but is not part of the reference transcript.
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
    """Create a process-owned lock so two local MPS benchmarks cannot overlap."""
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


def generate_corpus(directory: Path) -> list[dict[str, object]]:
    """Generate mono 16 kHz, 16-bit WAV files with the macOS speech synthesizer."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for item in CORPUS:
        path = directory / f"{item.id}.wav"
        command = [
            "say", "-v", item.voice, "-r", str(item.rate),
            "--data-format=LEI16@16000", "-o", str(path), item.say_text,
        ]
        subprocess.run(command, check=True)
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise RuntimeError(f"unexpected WAV format: {path}")
            sample_rate = source.getframerate()
            frames = source.getnframes()
        if sample_rate != 16_000:
            raise RuntimeError(f"unexpected sample rate {sample_rate}: {path}")
        record = asdict(item)
        record.update({
            "wav": str(path),
            "seconds": round(frames / sample_rate, 6),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        records.append(record)
    (directory / "manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    return records


def reuse_corpus(directory: Path) -> list[dict[str, object]]:
    """Load an existing corpus and reject missing, changed, or reordered WAVs."""
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
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != record.get("sha256"):
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
    """Apply a documented case-and-punctuation-insensitive ASR normalization."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character if character.isalnum() else " " for character in value)
    return " ".join(value.split())


def error_counts(reference: str, hypothesis: str) -> dict[str, int | bool]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    normalized_reference = normalized_text(reference)
    normalized_hypothesis = normalized_text(hypothesis)
    normalized_ref_words = normalized_reference.split()
    normalized_hyp_words = normalized_hypothesis.split()
    normalized_reference_characters = normalized_reference.replace(" ", "")
    normalized_hypothesis_characters = normalized_hypothesis.replace(" ", "")
    return {
        "exact_match": hypothesis == reference,
        "raw_word_edits": edit_distance(ref_words, hyp_words),
        "raw_reference_words": len(ref_words),
        "raw_character_edits": edit_distance(reference, hypothesis),
        "raw_reference_characters": len(reference),
        "normalized_match": normalized_hypothesis == normalized_reference,
        "normalized_word_edits": edit_distance(normalized_ref_words, normalized_hyp_words),
        "normalized_reference_words": len(normalized_ref_words),
        "normalized_character_edits": edit_distance(
            normalized_reference_characters, normalized_hypothesis_characters
        ),
        "normalized_reference_characters": len(normalized_reference_characters),
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


def tail_loss(last_visible: str, final_text: str) -> dict[str, int | bool]:
    """Measure how much of final output was absent from the last stable preview."""
    visible_words = normalized_text(last_visible).split()
    final_words = normalized_text(final_text).split()
    common_words = 0
    for visible, final in zip(visible_words, final_words):
        if visible != final:
            break
        common_words += 1
    common_characters = 0
    normalized_visible = normalized_text(last_visible)
    normalized_final = normalized_text(final_text)
    for visible, final in zip(normalized_visible, normalized_final):
        if visible != final:
            break
        common_characters += 1
    return {
        "last_visible_is_final_prefix": (
            not normalized_visible or normalized_final.startswith(normalized_visible)
        ),
        "final_words": len(final_words),
        "common_prefix_words": common_words,
        "extra_or_wrong_visible_words": len(visible_words) - common_words,
        "missing_final_words_after_common_prefix": len(final_words) - common_words,
        "missing_final_characters_after_common_prefix": len(normalized_final) - common_characters,
        "last_visible_to_final_word_edits": edit_distance(visible_words, final_words),
        "last_visible_to_final_character_edits": edit_distance(normalized_visible, normalized_final),
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
    released_at: list[float] = []
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
            released_at.append(time.monotonic())
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
    raw_times: list[float] = []
    stable_times: list[float] = []
    raw_snapshots: list[str] = []
    stable_snapshots: list[str] = []
    longform._leaf_prompt = measured_leaf_prompt
    try:
        stream = speech.transcribe(
            audio=chunks(), sample_rate=sample_rate, timestamps="none", stream=True
        )
        for update in stream:
            if update.get("provisional") is False:
                continue
            text = normalize_text(str(update.get("text", "")))
            raw_times.append(time.monotonic())
            raw_snapshots.append(text)
            visible = stabilizer.push(text)
            if visible is not None:
                stable_times.append(time.monotonic())
                stable_snapshots.append(visible)
        result = stream.result()
    finally:
        longform._leaf_prompt = original_leaf_prompt
    finished = time.monotonic()
    if not released_at:
        raise RuntimeError("audio producer never reached release")

    final_text = normalize_text(str(result.get("text", "")))
    release_visible = next(
        (
            text for when, text in reversed(list(zip(stable_times, stable_snapshots)))
            if when <= released_at[0]
        ),
        "",
    )
    end_visible = stable_snapshots[-1] if stable_snapshots else ""
    source_seconds = (len(audio) + round(sample_rate * tail_ms / 1000.0)) / sample_rate
    decoded_seconds = sum(decoded_frames) / 16_000.0
    first_raw = next(
        (when for when, text in zip(raw_times, raw_snapshots) if text), None
    )
    return {
        "preset": asdict(preset),
        "first_raw_interim_ms": (
            round((first_raw - started) * 1000.0, 2) if first_raw else None
        ),
        "first_stable_interim_ms": (
            round((stable_times[0] - started) * 1000.0, 2) if stable_times else None
        ),
        "release_to_final_ms": round((finished - released_at[0]) * 1000.0, 2),
        "wall_ms": round((finished - started) * 1000.0, 2),
        "final_text": final_text,
        "last_visible_at_release": release_visible,
        "last_visible_interim": end_visible,
        "raw_churn": churn(raw_snapshots),
        "stable_churn": churn(stable_snapshots),
        "tail_loss": tail_loss(release_visible, final_text),
        "end_of_stream_tail_loss": tail_loss(end_visible, final_text),
        "compute": {
            "model_invocations": len(decoded_frames),
            "decoded_input_seconds": round(decoded_seconds, 4),
            "source_seconds_including_tail": round(source_seconds, 4),
            "audio_compute_amplification": round(decoded_seconds / source_seconds, 6),
            "decoded_frames_per_call": decoded_frames,
        },
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def aggregate(rows: list[dict[str, object]], preset_name: str) -> dict[str, object]:
    selected = [row for row in rows if row["preset"]["name"] == preset_name]  # type: ignore[index]
    counts = [row["errors"] for row in selected]

    def total(key: str) -> int:
        return sum(int(item[key]) for item in counts)  # type: ignore[index]

    def ratio(edits: str, units: str) -> float:
        denominator = total(units)
        return round(total(edits) / denominator, 6) if denominator else 0.0

    first_raw = [
        float(row["first_raw_interim_ms"])
        for row in selected if row["first_raw_interim_ms"] is not None
    ]
    first_stable = [
        float(row["first_stable_interim_ms"])
        for row in selected if row["first_stable_interim_ms"] is not None
    ]
    release_final = [float(row["release_to_final_ms"]) for row in selected]
    decoded = sum(float(row["compute"]["decoded_input_seconds"]) for row in selected)  # type: ignore[index]
    source = sum(float(row["compute"]["source_seconds_including_tail"]) for row in selected)  # type: ignore[index]
    return {
        "utterances": len(selected),
        "exact_matches": sum(bool(item["exact_match"]) for item in counts),  # type: ignore[index]
        "exact_match_rate": round(sum(bool(item["exact_match"]) for item in counts) / len(selected), 6),  # type: ignore[index]
        "normalized_matches": sum(bool(item["normalized_match"]) for item in counts),  # type: ignore[index]
        "normalized_match_rate": round(sum(bool(item["normalized_match"]) for item in counts) / len(selected), 6),  # type: ignore[index]
        "raw_wer": ratio("raw_word_edits", "raw_reference_words"),
        "raw_cer": ratio("raw_character_edits", "raw_reference_characters"),
        "normalized_wer": ratio("normalized_word_edits", "normalized_reference_words"),
        "normalized_cer": ratio("normalized_character_edits", "normalized_reference_characters"),
        "first_raw_interim_ms": {
            "available": len(first_raw),
            "coverage": round(len(first_raw) / len(selected), 6),
            "median": round(statistics.median(first_raw), 3) if first_raw else None,
            "p95": percentile(first_raw, 0.95),
            "mean": round(statistics.fmean(first_raw), 3) if first_raw else None,
        },
        "first_stable_interim_ms": {
            "available": len(first_stable),
            "coverage": round(len(first_stable) / len(selected), 6),
            "median": round(statistics.median(first_stable), 3) if first_stable else None,
            "p95": percentile(first_stable, 0.95),
            "mean": round(statistics.fmean(first_stable), 3) if first_stable else None,
        },
        "release_to_final_ms": {
            "median": round(statistics.median(release_final), 3),
            "p95": percentile(release_final, 0.95),
            "mean": round(statistics.fmean(release_final), 3),
        },
        "raw_churn_snapshot_count": sum(int(row["raw_churn"]["snapshot_count"]) for row in selected),  # type: ignore[index]
        "raw_churn_changed_transitions": sum(int(row["raw_churn"]["changed_transitions"]) for row in selected),  # type: ignore[index]
        "raw_churn_character_edits": sum(int(row["raw_churn"]["character_edits"]) for row in selected),  # type: ignore[index]
        "raw_churn_retracted_characters": sum(int(row["raw_churn"]["retracted_characters"]) for row in selected),  # type: ignore[index]
        "stable_churn_snapshot_count": sum(int(row["stable_churn"]["snapshot_count"]) for row in selected),  # type: ignore[index]
        "stable_churn_changed_transitions": sum(int(row["stable_churn"]["changed_transitions"]) for row in selected),  # type: ignore[index]
        "stable_churn_character_edits": sum(int(row["stable_churn"]["character_edits"]) for row in selected),  # type: ignore[index]
        "stable_churn_retracted_characters": sum(int(row["stable_churn"]["retracted_characters"]) for row in selected),  # type: ignore[index]
        "tail_loss_missing_words": sum(int(row["tail_loss"]["missing_final_words_after_common_prefix"]) for row in selected),  # type: ignore[index]
        "tail_loss_word_edits": sum(int(row["tail_loss"]["last_visible_to_final_word_edits"]) for row in selected),  # type: ignore[index]
        "tail_loss_utterances": sum(int(row["tail_loss"]["last_visible_to_final_word_edits"]) > 0 for row in selected),  # type: ignore[index]
        "tail_completion_rate": round(
            sum(int(row["tail_loss"]["common_prefix_words"]) for row in selected)
            / sum(int(row["tail_loss"]["final_words"]) for row in selected), 6
        ),  # type: ignore[index]
        "end_of_stream_tail_loss_missing_words": sum(int(row["end_of_stream_tail_loss"]["missing_final_words_after_common_prefix"]) for row in selected),  # type: ignore[index]
        "end_of_stream_tail_loss_word_edits": sum(int(row["end_of_stream_tail_loss"]["last_visible_to_final_word_edits"]) for row in selected),  # type: ignore[index]
        "end_of_stream_tail_loss_utterances": sum(int(row["end_of_stream_tail_loss"]["last_visible_to_final_word_edits"]) > 0 for row in selected),  # type: ignore[index]
        "end_of_stream_tail_completion_rate": round(
            sum(int(row["end_of_stream_tail_loss"]["common_prefix_words"]) for row in selected)
            / sum(int(row["end_of_stream_tail_loss"]["final_words"]) for row in selected), 6
        ),  # type: ignore[index]
        "model_invocations": sum(int(row["compute"]["model_invocations"]) for row in selected),  # type: ignore[index]
        "decoded_input_seconds": round(decoded, 4),
        "source_seconds_including_tail": round(source, 4),
        "audio_compute_amplification": round(decoded / source, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--tail-ms", type=float, default=320.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--reuse-corpus", action="store_true",
        help="use and hash-check the existing manifest and WAVs instead of regenerating them",
    )
    args = parser.parse_args()
    if args.frame_ms <= 0 or args.tail_ms < 0:
        parser.error("--frame-ms must be positive and --tail-ms must be non-negative")

    manifest = (
        reuse_corpus(args.corpus_dir) if args.reuse_corpus else generate_corpus(args.corpus_dir)
    )
    if args.generate_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

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
        try:
            photon, speech = load_model("mps", announce=False)
            for item_index, item in enumerate(CORPUS):
                audio, sample_rate = read_wav(args.corpus_dir / f"{item.id}.wav")
                rotation = item_index % len(PRESETS)
                order = PRESETS[rotation:] + PRESETS[:rotation]
                for order_index, preset in enumerate(order):
                    row = run_preset(
                        speech, audio, sample_rate, preset,
                        frame_ms=args.frame_ms, tail_ms=args.tail_ms,
                    )
                    row.update({
                        "utterance": asdict(item),
                        "audio_seconds": round(len(audio) / sample_rate, 6),
                        "order_in_utterance": order_index + 1,
                    })
                    row["errors"] = error_counts(item.reference, str(row["final_text"]))
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

    baseline_by_utterance = {
        str(row["utterance"]["id"]): str(row["final_text"])
        for row in rows
        if row["preset"]["name"] == PRESETS[0].name
    }
    for row in rows:
        baseline = baseline_by_utterance[str(row["utterance"]["id"])]
        row["final_equals_baseline"] = row["final_text"] == baseline
        row["normalized_final_equals_baseline"] = (
            normalized_text(str(row["final_text"])) == normalized_text(baseline)
        )

    aggregates = {preset.name: aggregate(rows, preset.name) for preset in PRESETS}
    for preset in PRESETS:
        selected = [row for row in rows if row["preset"]["name"] == preset.name]
        aggregates[preset.name]["final_parity_with_baseline"] = {
            "exact": sum(bool(row["final_equals_baseline"]) for row in selected),
            "normalized": sum(bool(row["normalized_final_equals_baseline"]) for row in selected),
            "utterances": len(selected),
        }
    report = {
        "experiment": "first-visible-latency-quality-production-stabilizer-apple-mps",
        "versions": {
            "kestrel": version("kestrel"),
            "kestrel-kernels": version("kestrel-kernels"),
            "torch": version("torch"),
        },
        "platform": platform.platform(),
        "macos_product_version": subprocess.run(
            ["sw_vers", "-productVersion"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "say_recipe": "say -v VOICE -r RATE --data-format=LEI16@16000 -o WAV SAY_TEXT",
        "corpus_dir": str(args.corpus_dir),
        "corpus_reused": args.reuse_corpus,
        "corpus": manifest,
        "frame_ms": args.frame_ms,
        "tail_ms": args.tail_ms,
        "realtime_feed": True,
        "rotating_preset_order": True,
        "order_schedule": "ABC, BCA, CAB repeated across the 12 corpus utterances",
        "presets": [asdict(preset) for preset in PRESETS],
        "runs": rows,
        "aggregates": aggregates,
        "metric_notes": [
            "Raw WER/CER compare whitespace tokens/code points exactly, including case and punctuation.",
            "Normalized text uses Unicode NFKC, casefolding, punctuation-to-space, and whitespace collapse; it does not normalize spoken numbers. Normalized CER excludes spaces.",
            "Corpus WER/CER are micro-averages: summed edit distance divided by summed reference units.",
            "Tail loss compares the last stable interim with release-final output after normalization.",
            "Amplification is summed decoded input seconds divided by source seconds including the 320 ms synthetic tail.",
            "Preset order rotates ABC, BCA, CAB by utterance; one warm model is shared; the MPS lock covers model load and all runs.",
            "First raw is the first non-empty provisional model update; first stable is the first non-empty value emitted by the production InterimTranscriptStabilizer.",
            "Stability 1 still applies the production complete-word-boundary rule; it is not raw passthrough.",
        ],
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
