#!/usr/bin/env python3
"""Benchmark isolated Kestrel 0.8 MPS streaming readback reductions.

The candidate is a process-local replacement for the streaming result assembly.
It does not modify Kestrel, the worker, or the installed environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
import traceback
import wave
from typing import Any, Callable, Iterator, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCK_PATH = Path("/tmp/ptt-mps-benchmark.lock")
EXPECTED = {"torch": "2.14.0", "kestrel": "0.8.0", "kestrel-kernels": "0.7.0"}


@contextlib.contextmanager
def atomic_mps_lock() -> Iterator[None]:
    try:
        fd = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        owner = LOCK_PATH.read_text(errors="replace").strip()
        raise RuntimeError(f"MPS benchmark lock is held ({LOCK_PATH}: {owner})") from exc
    try:
        os.write(fd, f"pid={os.getpid()} script={Path(__file__).resolve()}\n".encode())
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


class HostReadbackProbe:
    """Count Python-visible MPS materializations and explicit synchronizations."""

    METHODS = ("tolist", "item", "cpu", "numpy", "__int__", "__float__", "__bool__")

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.phase = "idle"
        self.events: Counter[tuple[str, str, str]] = Counter()
        self.syncs: Counter[tuple[str, str]] = Counter()
        self._originals: dict[str, Callable[..., Any]] = {}
        self._original_sync: Callable[..., Any] | None = None
        self._mutex = threading.Lock()

    @staticmethod
    def _site() -> str:
        frames = traceback.extract_stack(limit=20)[:-2]
        for frame in reversed(frames):
            normalized = frame.filename.replace("\\", "/")
            if "/kestrel/" in normalized:
                return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        for frame in reversed(frames):
            if (
                frame.filename.endswith("benchmark-mps-readback.py")
                and frame.name not in {"wrapped", "_record", "_site"}
            ):
                return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
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

    def __enter__(self) -> HostReadbackProbe:
        for name in self.METHODS:
            original = getattr(self.torch.Tensor, name)
            self._originals[name] = original

            def wrapped(tensor: Any, *args: Any, __name: str = name, __original: Any = original, **kwargs: Any) -> Any:
                self._record(f"Tensor.{__name}", tensor)
                return __original(tensor, *args, **kwargs)

            setattr(self.torch.Tensor, name, wrapped)
        self._original_sync = self.torch.mps.synchronize

        def synchronize(*args: Any, **kwargs: Any) -> Any:
            with self._mutex:
                self.syncs[(self.phase, self._site())] += 1
            assert self._original_sync is not None
            return self._original_sync(*args, **kwargs)

        self.torch.mps.synchronize = synchronize
        return self

    def __exit__(self, *_args: Any) -> None:
        for name, original in self._originals.items():
            setattr(self.torch.Tensor, name, original)
        if self._original_sync is not None:
            self.torch.mps.synchronize = self._original_sync

    def snapshot(self, phase: str) -> dict[str, Any]:
        materializations = [
            {"kind": kind, "site": site, "count": count}
            for (event_phase, kind, site), count in sorted(self.events.items())
            if event_phase == phase
        ]
        syncs = [
            {"site": site, "count": count}
            for (sync_phase, site), count in sorted(self.syncs.items())
            if sync_phase == phase
        ]
        return {
            "python_visible_mps_materializations": sum(row["count"] for row in materializations),
            "explicit_torch_mps_synchronize_calls": sum(row["count"] for row in syncs),
            "materialization_sites": materializations,
            "explicit_sync_sites": syncs,
        }


@dataclass(frozen=True, slots=True)
class DeviceStreamState:
    decoder: Any
    token_ids: Any
    durations: Any


@dataclass
class RunRecorder:
    phase: str
    preview_decode_ms: list[float]
    token_snapshots: list[list[int]]
    duration_snapshots: list[list[int]]
    native_decision_readbacks: int = 0


def _host_tokens(state: Any) -> tuple[list[int], list[int]]:
    tokens = state.token_ids
    durations = state.durations
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if hasattr(durations, "tolist"):
        durations = durations.tolist()
    return list(tokens), list(durations)


def stock_recording_method(original: Callable[..., Any], recorder: RunRecorder) -> Callable[..., Any]:
    def wrapped(runtime: Any, rows: Any, windows: Any, requests: Any, features: Any, mask: Any, *, max_tokens: int) -> Any:
        started = time.perf_counter()
        values = original(runtime, rows, windows, requests, features, mask, max_tokens=max_tokens)
        recorder.preview_decode_ms.append((time.perf_counter() - started) * 1000.0)
        for value, window in zip(values, windows, strict=True):
            state = value["_stream_state"]
            tokens, durations = _host_tokens(state)
            prior = 1 if window.state is None else len(window.state.token_ids)
            recorder.native_decision_readbacks += len(tokens) - prior
            recorder.token_snapshots.append(tokens)
            recorder.duration_snapshots.append(durations)
        return values

    return wrapped



def generate_encoded_with_length(
    model: Any,
    encoded: Any,
    valid_length: int,
    *,
    max_tokens: int | None,
    start_frame: int,
    frame_count: int,
    state: Any,
) -> Any:
    """Kestrel 0.8 single-row decoder with an exact host-derived length."""
    import torch
    from kestrel.models.parakeet_tdt.model import TdtOutput, TdtState
    from kestrel_kernels import get_runtime

    end_frame = min(valid_length, start_frame + frame_count)
    if not 0 <= start_frame <= end_frame:
        raise ValueError("TDT decode frames are outside the encoded audio")
    token = torch.tensor([[model.config.blank_token_id]], device=encoded.device)
    if state is None:
        decoder_hidden, decoder_state = model.decoder(token, None)
        carry = 0
    else:
        decoder_hidden = state.decoder_hidden
        decoder_state = (state.hidden, state.cell)
        carry = state.carry
    frame = start_frame + carry
    sequences = [model.config.blank_token_id]
    durations = [min(carry, end_frame - start_frame)]
    steps_remaining = model.config.max_symbols_per_step * (end_frame - start_frame)
    tokens_remaining = max_tokens
    greedy_step = get_runtime().tdt.greedy_step
    while frame < end_frame and steps_remaining > 0 and (tokens_remaining is None or tokens_remaining > 0):
        logits = model.joint(encoded[:, frame : frame + 1], decoder_hidden)
        token_id, duration_index = greedy_step(logits, model.config.vocab_size)
        duration = model.config.durations[duration_index]
        if token_id == model.config.blank_token_id and duration == 0:
            duration = 1
        sequences.append(token_id)
        durations.append(duration)
        frame += duration
        if token_id != model.config.blank_token_id:
            token.fill_(token_id)
            decoder_hidden, decoder_state = model.decoder(token, decoder_state)
            if tokens_remaining is not None:
                tokens_remaining -= 1
        steps_remaining -= 1
    hidden, cell = decoder_state
    return TdtOutput(
        sequences=torch.tensor([sequences], dtype=torch.long, device=encoded.device),
        durations=torch.tensor([durations], dtype=torch.long, device=encoded.device),
        lengths=torch.tensor([len(sequences)], device=encoded.device),
        state=TdtState(decoder_hidden, hidden, cell, max(0, frame - end_frame)),
        encoder_frame_seconds=model.encoder_frame_seconds,
    )

def candidate_method(recorder: RunRecorder) -> Callable[..., Any]:
    """Keep cumulative tokens on MPS and use host-derived valid lengths."""
    import torch
    from kestrel.models.asr.audio import DecodedAudio
    from kestrel.models.asr.contract import Segment, TranscriptionResult
    from kestrel.models.parakeet_tdt.runtime import _encoder_frames

    def run(runtime: Any, rows: Sequence[Any], windows: Sequence[Any], requests: Sequence[Any], features: Any, mask: Any, *, max_tokens: int) -> tuple[dict[str, object], ...]:
        started = time.perf_counter()
        values: list[dict[str, object]] = []
        factor = runtime.model.config.encoder.subsampling_factor
        with runtime._encoder_graph.launch(features, mask) as (encoded, _valid):
            for row, window, request, row_encoded in zip(rows, windows, requests, encoded, strict=True):
                _index, audio = row
                previous = window.state
                # parakeet_features marks samples // 160 mel frames valid. Each
                # stride-2 subsampler layer applies ceil(length / 2), exactly the
                # same integer recurrence as _encoder_frames.
                valid_length = _encoder_frames(audio.waveform.size, factor)
                generated = generate_encoded_with_length(
                    runtime.model,
                    row_encoded[None],
                    valid_length,
                    max_tokens=max_tokens,
                    start_frame=_encoder_frames(window.start_sample, factor),
                    frame_count=_encoder_frames(window.sample_count, factor),
                    state=None if previous is None else previous.decoder,
                )
                if generated.state is None:
                    raise RuntimeError("stateful TDT decoding returned no state")
                new_tokens = generated.sequences[0, 1:]
                new_durations = generated.durations[0, 1:]
                recorder.native_decision_readbacks += new_tokens.shape[0]
                if previous is None:
                    prefix_tokens = torch.tensor([runtime.tokenizer.blank_token_id], device=encoded.device)
                    prefix_durations = torch.zeros(1, dtype=torch.long, device=encoded.device)
                else:
                    prefix_tokens = previous.token_ids
                    prefix_durations = previous.durations
                all_tokens = torch.cat((prefix_tokens, new_tokens))
                all_durations = torch.cat((prefix_durations, new_durations))
                state = DeviceStreamState(generated.state, all_tokens, all_durations)

                # This is the only token/duration materialization. It occurs at
                # the runtime boundary which immediately supplies the UI preview.
                packed = torch.stack((all_tokens, all_durations), dim=1).tolist()
                token_ids = [pair[0] for pair in packed]
                durations = [pair[1] for pair in packed]
                recorder.token_snapshots.append(token_ids)
                recorder.duration_snapshots.append(durations)
                text_parts: list[str] = []
                segments: list[Segment] = []
                logical_audio = DecodedAudio(
                    audio.waveform,
                    window.duration_seconds,
                    window.duration_seconds,
                    0.0,
                )
                runtime._append_chunk_result(
                    request,
                    logical_audio,
                    token_ids,
                    durations,
                    text_parts,
                    segments,
                    frame_seconds=generated.encoder_frame_seconds,
                )
                value = TranscriptionResult(
                    text=" ".join(text_parts),
                    language=None,
                    duration_seconds=window.duration_seconds,
                    source_duration_seconds=window.duration_seconds,
                    clip_start_seconds=0.0,
                    segments=tuple(segments),
                ).as_dict()
                value["_stream_state"] = state
                values.append(value)
        recorder.preview_decode_ms.append((time.perf_counter() - started) * 1000.0)
        return tuple(values)

    return run


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "max_ms": round(max(values), 3),
        "total_ms": round(sum(values), 3),
    }


def run_stream(
    speech: Any, audio: np.ndarray, rate: int, chunk_ms: int
) -> tuple[str, list[str], float, float]:
    frames = max(1, round(rate * chunk_ms / 1000.0))
    released_at = 0.0

    async def chunks():
        nonlocal released_at
        for start in range(0, audio.size, frames):
            if start + frames >= audio.size:
                released_at = time.perf_counter()
            yield audio[start : start + frames]

    started = time.perf_counter()
    stream = speech.transcribe(audio=chunks(), sample_rate=rate, timestamps="none", stream=True)
    previews = [str(update.get("text", "")) for update in stream]
    result = stream.result()
    ended = time.perf_counter()
    if not released_at:
        raise RuntimeError("stream did not consume the final audio chunk")
    return (
        str(result.get("text", "")),
        previews,
        (ended - started) * 1000.0,
        (ended - released_at) * 1000.0,
    )


def run_mode(speech: Any, audio: np.ndarray, rate: int, chunk_ms: int, mode: str, probe: HostReadbackProbe) -> dict[str, Any]:
    import kestrel.models.parakeet_tdt.runtime as runtime_module

    recorder = RunRecorder(mode, [], [], [])
    original = runtime_module.ParakeetTdtRuntime._run_stream_group
    replacement = stock_recording_method(original, recorder) if mode == "stock" else candidate_method(recorder)
    runtime_module.ParakeetTdtRuntime._run_stream_group = replacement
    probe.phase = mode
    before = probe.snapshot(mode)
    try:
        text, previews, total_ms, release_final_ms = run_stream(
            speech, audio, rate, chunk_ms
        )
    finally:
        runtime_module.ParakeetTdtRuntime._run_stream_group = original
    after = probe.snapshot(mode)
    materializations = (
        after["python_visible_mps_materializations"]
        - before["python_visible_mps_materializations"]
    )
    explicit_syncs = (
        after["explicit_torch_mps_synchronize_calls"]
        - before["explicit_torch_mps_synchronize_calls"]
    )
    return {
        "mode": mode,
        "final_text": text,
        "preview_texts": previews,
        "token_snapshots": recorder.token_snapshots,
        "duration_snapshots": recorder.duration_snapshots,
        "preview_decode_latency": summary(recorder.preview_decode_ms),
        "preview_decode_latency_samples_ms": recorder.preview_decode_ms,
        "release_final_ms": round(release_final_ms, 3),
        "end_to_end_ms": round(total_ms, 3),
        "sync_count": {
            "python_visible_mps_materializations": materializations,
            "explicit_torch_mps_synchronize_calls": explicit_syncs,
            "native_tdt_decision_readbacks": recorder.native_decision_readbacks,
            "python_visible_host_wait_lower_bound": materializations + explicit_syncs,
        },
    }


def _parity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    return {
        "preview_transcript_sequence": left["preview_texts"] == right["preview_texts"],
        "final_transcript": left["final_text"] == right["final_text"],
        "token_ids": left["token_snapshots"] == right["token_snapshots"],
        "durations": left["duration_snapshots"] == right["duration_snapshots"],
    }


def _totals(runs: list[dict[str, Any]], key: str) -> int:
    return sum(int(run["sync_count"][key]) for run in runs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="directory containing manifest.json for an alternating corpus run",
    )
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if args.wav is not None and args.corpus is not None:
        parser.error("wav and --corpus are mutually exclusive")

    if args.corpus is not None:
        manifest_path = args.corpus / "manifest.json"
        inputs = json.loads(manifest_path.read_text())
    else:
        wav = args.wav or Path("assets/demo-input.wav")
        inputs = [{"id": wav.stem, "wav": str(wav)}]

    failures: list[dict[str, str]] = []
    prompt_reports: list[dict[str, Any]] = []
    diagnostic_runs: dict[str, list[dict[str, Any]]] = {
        "stock": [],
        "candidate": [],
    }
    timing_runs: dict[str, list[dict[str, Any]]] = {
        "stock": [],
        "candidate": [],
    }

    with atomic_mps_lock():
        import torch

        versions = {name: version(name) for name in EXPECTED}
        if versions != EXPECTED:
            raise RuntimeError(f"version mismatch: actual={versions}, expected={EXPECTED}")
        if sys.platform != "darwin" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS on macOS arm64 is required")
        loaded = [(item, *read_wav(Path(item["wav"]))) for item in inputs]
        os.environ["PTT_SKIP_MODEL_WARMUP"] = "1"
        from ptt_worker import load_model

        photon = None
        probe = HostReadbackProbe(torch)
        try:
            photon, speech = load_model("mps", announce=False)
            first_audio, first_rate = loaded[0][1], loaded[0][2]
            speech.transcribe(
                audio=first_audio[: min(first_audio.size, first_rate)],
                sample_rate=first_rate,
                timestamps="none",
            )
            for index, (item, audio, rate) in enumerate(loaded):
                first = "stock" if index % 2 == 0 else "candidate"
                order = [first, "candidate" if first == "stock" else "stock"]
                try:
                    diagnostic: dict[str, dict[str, Any]] = {}
                    with probe:
                        for mode in order:
                            diagnostic[mode] = run_mode(
                                speech, audio, rate, args.chunk_ms, mode, probe
                            )
                            diagnostic_runs[mode].append(diagnostic[mode])
                    timed: dict[str, dict[str, Any]] = {}
                    for mode in order:
                        timed[mode] = run_mode(
                            speech, audio, rate, args.chunk_ms, mode, probe
                        )
                        timing_runs[mode].append(timed[mode])

                    diagnostic_parity = _parity(
                        diagnostic["stock"], diagnostic["candidate"]
                    )
                    timing_parity = _parity(timed["stock"], timed["candidate"])
                    passed = all(diagnostic_parity.values()) and all(timing_parity.values())
                    if not passed:
                        failures.append(
                            {
                                "id": str(item["id"]),
                                "error": (
                                    f"parity mismatch: diagnostic={diagnostic_parity}, "
                                    f"timing={timing_parity}"
                                ),
                            }
                        )
                    prompt_reports.append(
                        {
                            "id": item["id"],
                            "wav": item["wav"],
                            "order": order,
                            "samples": int(audio.size),
                            "sample_rate": rate,
                            "parity": diagnostic_parity,
                            "probe_off_repeat_parity": timing_parity,
                            "passed": passed,
                            "stock": diagnostic["stock"],
                            "candidate": diagnostic["candidate"],
                            "timing": {
                                mode: {
                                    "preview_decode_latency": timed[mode]["preview_decode_latency"],
                                    "release_final_ms": timed[mode]["release_final_ms"],
                                    "end_to_end_ms": timed[mode]["end_to_end_ms"],
                                }
                                for mode in ("stock", "candidate")
                            },
                        }
                    )
                except Exception as exc:  # Continue so the report lists all failures.
                    failures.append(
                        {"id": str(item.get("id", index)), "error": repr(exc)}
                    )
                    prompt_reports.append(
                        {
                            "id": item.get("id", index),
                            "wav": item["wav"],
                            "order": order,
                            "passed": False,
                            "error": repr(exc),
                        }
                    )
        finally:
            if photon is not None:
                with contextlib.redirect_stdout(None):
                    photon.__exit__(None, None, None)

        metrics: dict[str, dict[str, Any]] = {}
        for mode in ("stock", "candidate"):
            preview_samples = [
                value
                for run in timing_runs[mode]
                for value in run["preview_decode_latency_samples_ms"]
            ]
            metrics[mode] = {
                "preview_decode_ms": summary(preview_samples),
                "release_final_ms": summary(
                    [run["release_final_ms"] for run in timing_runs[mode]]
                ),
                "end_to_end_ms": summary(
                    [run["end_to_end_ms"] for run in timing_runs[mode]]
                ),
                "materialization_counts": {
                    "python_visible_mps_materializations": _totals(
                        diagnostic_runs[mode], "python_visible_mps_materializations"
                    ),
                    "explicit_torch_mps_synchronize_calls": _totals(
                        diagnostic_runs[mode], "explicit_torch_mps_synchronize_calls"
                    ),
                    "native_tdt_decision_readbacks": _totals(
                        diagnostic_runs[mode], "native_tdt_decision_readbacks"
                    ),
                    "python_visible_host_wait_lower_bound": _totals(
                        diagnostic_runs[mode], "python_visible_host_wait_lower_bound"
                    ),
                },
            }

        report = {
            "experiment": "kestrel-0.8-apple-mps-streaming-readback-corpus",
            "environment": {"platform": platform.platform(), "versions": versions},
            "input": {
                "corpus": str(args.corpus) if args.corpus else None,
                "prompt_count": len(inputs),
                "chunk_ms": args.chunk_ms,
            },
            "method": {
                "order": "Alternating first mode by prompt; same order for diagnostic and probe-off timing repeats.",
                "lock": str(LOCK_PATH),
                "latency_probe": "off",
            },
            "all_parity_passed": not failures and len(prompt_reports) == len(inputs),
            "failure_count": len(failures),
            "failures": failures,
            "metrics": metrics,
            "prompts": prompt_reports,
            "scope": [
                "Process-local monkey patch only; production and .venv are unchanged.",
                "Exact parity is literal and covers the ordered preview transcript sequence, final transcript, cumulative token IDs, and cumulative durations for both diagnostic and probe-off repeats.",
                "Python-visible materialization counts exclude waits hidden inside native extensions.",
                "Latency comes from probe-off repeats; release-final starts when the final audio chunk is yielded.",
            ],
        }
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        print(rendered)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
