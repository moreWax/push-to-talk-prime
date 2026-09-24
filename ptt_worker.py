#!/usr/bin/env python3
"""Persistent stdin/stdout JSON-lines worker for local Parakeet transcription."""

from __future__ import annotations

import contextlib
from collections import deque
import json
import os
from importlib.metadata import PackageNotFoundError, version
import queue
import signal
import sys
import threading
import time
import wave
from types import SimpleNamespace
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

_PROTOCOL_OUT = sys.stdout
_EMIT_LOCK = threading.Lock()


def emit(payload: dict[str, Any]) -> None:
    with _EMIT_LOCK:
        _PROTOCOL_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _PROTOCOL_OUT.flush()


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


class DemoAudioInputStream:
    """Real-time WAV-backed input used only by the reproducible demo harness."""

    def __init__(self, path: str, callback: Any, np: Any) -> None:
        with wave.open(path, "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("PTT_DEMO_AUDIO_FILE must be mono 16-bit PCM WAV")
            self.sample_rate = source.getframerate()
            self.samples = np.frombuffer(
                source.readframes(source.getnframes()),
                dtype="<i2",
            ).astype(np.float32) / 32768.0
        self.callback = callback
        self.np = np
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return

        def run() -> None:
            block_frames = max(1, round(self.sample_rate * 0.02))
            started = time.monotonic()
            offset = 0
            while not self.stop_event.is_set():
                target = started + offset / self.sample_rate
                if self.stop_event.wait(max(0.0, target - time.monotonic())):
                    break
                if offset < len(self.samples):
                    block = self.samples[offset : offset + block_frames]
                else:
                    block = self.np.zeros(block_frames, dtype=self.np.float32)
                if len(block) < block_frames:
                    block = self.np.pad(block, (0, block_frames - len(block)))
                self.callback(
                    block[:, None],
                    len(block),
                    SimpleNamespace(currentTime=time.monotonic()),
                    None,
                )
                offset += block_frames

        self.thread = threading.Thread(target=run, name="ptt-demo-audio", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()


def select_capture_sample_rate(sd: Any, device: int | str | None, native_rate: int) -> tuple[int, bool]:
    """Prefer model-native 16 kHz capture, with a device-native fallback."""
    raw = os.getenv("PTT_CAPTURE_SAMPLE_RATE", "16000").strip().lower()
    if raw in {"native", "default"}:
        return native_rate, False
    try:
        requested = int(raw)
    except ValueError as exc:
        raise ValueError("PTT_CAPTURE_SAMPLE_RATE must be an integer or 'native'") from exc
    if requested <= 0:
        raise ValueError("PTT_CAPTURE_SAMPLE_RATE must be positive")
    if requested == native_rate:
        return native_rate, False
    try:
        sd.check_input_settings(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=requested,
        )
    except Exception:
        return native_rate, True
    return requested, False


class Recorder:
    def __init__(self, sd: Any, np: Any) -> None:
        self.sd = sd
        self.np = np
        self.stream: Any | None = None
        self.frames: list[Any] = []
        self.lock = threading.Lock()
        self.sample_rate = 16_000
        self.native_sample_rate = 16_000
        self.capture_rate_fallback = False
        self.last_level_emit = 0.0
        self.live_queue: queue.Queue[Any | None] | None = None

    def start(self, live_queue: queue.Queue[Any | None] | None = None) -> None:
        if self.stream is not None:
            raise RuntimeError("microphone is already recording")
        demo_audio = os.getenv("PTT_DEMO_AUDIO_FILE")
        device_env = os.getenv("PTT_INPUT_DEVICE")
        device: int | str | None = None
        if device_env:
            device = int(device_env) if device_env.isdigit() else device_env
        if demo_audio:
            with wave.open(demo_audio, "rb") as source:
                self.sample_rate = source.getframerate()
            self.native_sample_rate = self.sample_rate
            self.capture_rate_fallback = False
        else:
            info = self.sd.query_devices(device, "input")
            self.native_sample_rate = int(info["default_samplerate"])
            self.sample_rate, self.capture_rate_fallback = select_capture_sample_rate(
                self.sd,
                device,
                self.native_sample_rate,
            )
        self.frames = []
        self.live_queue = live_queue

        def callback(indata: Any, _frames: int, time_info: Any, status: Any) -> None:
            if status:
                print(f"audio warning: {status}", file=sys.stderr)
            mono = indata[:, 0].copy()
            with self.lock:
                self.frames.append(mono)
            if self.live_queue is not None:
                self.live_queue.put(mono)
            # Match Claude Code's perceptual level curve closely. Limit updates
            # to about 20 fps so UI rendering never delays audio capture.
            now = float(getattr(time_info, "currentTime", 0.0) or 0.0)
            if now <= 0:
                import time
                now = time.monotonic()
            if now - self.last_level_emit >= 0.05:
                self.last_level_emit = now
                rms = float(self.np.sqrt(self.np.mean(mono * mono))) if len(mono) else 0.0
                level = float(self.np.sqrt(min(rms * 16.384, 1.0)))
                emit({"event": "level", "level": level})

        if demo_audio:
            self.stream = DemoAudioInputStream(demo_audio, callback, self.np)
            self.stream.start()
            return

        def open_stream(sample_rate: int) -> Any:
            return self.sd.InputStream(
                device=device,
                channels=1,
                samplerate=sample_rate,
                dtype="float32",
                callback=callback,
            )

        def start_stream(sample_rate: int) -> Any:
            stream = open_stream(sample_rate)
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            return stream

        try:
            self.stream = start_stream(self.sample_rate)
        except Exception:
            self.stream = None
            if self.sample_rate == self.native_sample_rate:
                raise
            self.sample_rate = self.native_sample_rate
            self.capture_rate_fallback = True
            try:
                self.stream = start_stream(self.sample_rate)
            except Exception:
                self.stream = None
                raise

    def finish(self) -> tuple[Any, int]:
        if self.stream is None:
            raise RuntimeError("microphone is not recording")
        stream, self.stream = self.stream, None
        chunks: list[Any] = []
        try:
            try:
                stream.stop()
            finally:
                stream.close()
        finally:
            if self.live_queue is not None:
                tail_ms = max(0.0, float(os.getenv("PTT_TRAILING_SILENCE_MS", "320")))
                tail_frames = round(self.sample_rate * tail_ms / 1000.0)
                if tail_frames > 0:
                    self.live_queue.put(self.np.zeros(tail_frames, dtype=self.np.float32))
                self.live_queue.put(None)
                self.live_queue = None
            with self.lock:
                chunks, self.frames = self.frames, []
        if not chunks:
            return self.np.empty(0, dtype=self.np.float32), self.sample_rate
        return self.np.concatenate(chunks), self.sample_rate

    def cancel(self) -> None:
        if self.stream is None:
            return
        stream, self.stream = self.stream, None
        try:
            try:
                stream.stop()
            finally:
                stream.close()
        finally:
            if self.live_queue is not None:
                self.live_queue.put(None)
                self.live_queue = None
            with self.lock:
                self.frames = []


class _IntegralFrameSeconds(float):
    """Fractional seconds that produce integral frame counts in Kestrel 0.8."""

    def __mul__(self, value: object) -> object:
        if isinstance(value, (int, float)):
            return round(float(self) * value)
        return NotImplemented

    def __rmul__(self, value: object) -> object:
        return self.__mul__(value)


def warm_streaming_model(speech: Any, chunk_ms: float, right_ms: float) -> None:
    if os.getenv("PTT_SKIP_MODEL_WARMUP") == "1":
        return
    import numpy as np

    sample_rate = 16_000
    total_frames = max(320, round(sample_rate * (chunk_ms + right_ms) / 1000.0))

    async def silence():
        yield np.zeros(total_frames, dtype=np.float32)

    stream = speech.transcribe(
        audio=silence(),
        sample_rate=sample_rate,
        timestamps="none",
        stream=True,
    )
    for _update in stream:
        pass
    stream.result()


def load_model(requested: str, *, announce: bool = True) -> tuple[Any, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        import moondream as md
        import kestrel.engine.core as kestrel_core
        import kestrel.models.parakeet_tdt.longform as parakeet_longform

        if os.getenv("PTT_ALLOW_TELEMETRY") != "1":
            class _NoTelemetryReporter:
                def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                    pass

                @staticmethod
                def _normalize_api_key(_value: object) -> None:
                    return None

                @staticmethod
                def _is_api_key_header_safe(_value: object) -> bool:
                    return False

                async def validate_api_key(self) -> bool:
                    return False

                def start(self) -> None:
                    pass

                async def shutdown(self) -> None:
                    pass

                def record_success(self, **_kwargs: Any) -> None:
                    pass

                def record_error(self, **_kwargs: Any) -> None:
                    pass

            kestrel_core.PhotonReporter = _NoTelemetryReporter

        kestrel_version = version("kestrel")
        if kestrel_version != "0.8.0":
            raise RuntimeError(
                f"unsupported Kestrel version {kestrel_version}; expected 0.8.0 for streaming window compatibility"
            )

        # Kestrel 0.8 defaults to 2 s preview chunks plus 2 s of right context.
        # Parakeet's stateful TDT path supports smaller increments. Empirical
        # validation shows asymmetric chunk/lookahead windows can preserve stable
        # hypotheses while increasing update cadence. The default is 160/480.
        chunk_ms = max(40.0, float(os.getenv("PTT_STREAM_CHUNK_MS", "160")))
        right_ms = max(0.0, float(os.getenv("PTT_STREAM_RIGHT_MS", "480")))
        left_ms = max(chunk_ms, float(os.getenv("PTT_STREAM_LEFT_MS", "4000")))
        parakeet_longform._LIVE_CHUNK_SECONDS = _IntegralFrameSeconds(chunk_ms / 1000.0)
        parakeet_longform._LIVE_RIGHT_SECONDS = _IntegralFrameSeconds(right_ms / 1000.0)
        parakeet_longform._LIVE_LEFT_SECONDS = _IntegralFrameSeconds(left_ms / 1000.0)

        kwargs = {} if requested == "auto" else {"device": requested}
        photon = md.photon("moondream/parakeet-redux", **kwargs)
        speech = photon.__enter__()
        warm_streaming_model(speech, chunk_ms, right_ms)
    if announce:
        emit({
            "event": "model_ready",
            "model": "moondream/parakeet-redux",
            "device": requested,
            "stream_chunk_ms": chunk_ms,
            "stream_right_ms": right_ms,
            "stream_left_ms": left_ms,
        })
    return photon, speech


class InterimTranscriptStabilizer:
    """Expose only text that survives consecutive cumulative ASR snapshots."""

    def __init__(self, required_snapshots: int = 2) -> None:
        self.required_snapshots = max(1, required_snapshots)
        self.history: deque[str] = deque(maxlen=self.required_snapshots)
        self.visible = ""

    def push(self, text: str) -> str | None:
        value = normalize_text(text)
        if not value:
            return None
        self.history.append(value)
        if len(self.history) < self.required_snapshots:
            return None

        values = list(self.history)
        prefix = values[0]
        for item in values[1:]:
            limit = min(len(prefix), len(item))
            index = 0
            while index < limit and prefix[index] == item[index]:
                index += 1
            prefix = prefix[:index]
            if not prefix:
                break

        if not prefix:
            candidate = ""
        elif all(item == prefix for item in values):
            # Equality alone does not prove that the trailing SentencePiece
            # fragment is a complete word. Keep only text through the last
            # visible boundary; punctuation itself is safe to show.
            punctuation = max(*(prefix.rfind(char) for char in ".,!?;:"))
            whitespace = prefix.rfind(" ")
            boundary = max(punctuation, whitespace)
            candidate = prefix[: boundary + 1] if boundary >= 0 else ""
        else:
            boundary_after_prefix = all(
                len(item) == len(prefix)
                or item[len(prefix)].isspace()
                or item[len(prefix)] in ".,!?;:"
                for item in values
            )
            if boundary_after_prefix:
                candidate = prefix
            else:
                boundary = max(prefix.rfind(" "), *(prefix.rfind(char) for char in ".,!?;:"))
                candidate = prefix[: boundary + 1] if boundary >= 0 else ""

        candidate = candidate.strip()
        if candidate == self.visible:
            return None
        self.visible = candidate
        return candidate or None


def transcribe_live(
    model_future: "Future[tuple[Any, Any]]",
    chunks: "queue.Queue[Any | None]",
    sample_rate: int,
    session_id: int,
    is_current: Any,
) -> dict[str, object]:
    _photon, speech = model_future.result(timeout=300)
    early_draft_ms = max(0.0, float(os.getenv("PTT_EARLY_DRAFT_MS", "0")))
    early_draft_frames = round(sample_rate * early_draft_ms / 1000.0)

    def transcribe_draft(audio: Any) -> str:
        try:
            result = speech.transcribe(
                audio=audio,
                sample_rate=sample_rate,
                timestamps="none",
                stream=False,
            )
            if not isinstance(result, dict):
                return ""
            return normalize_text(str(result.get("text", "")))
        except Exception as exc:
            print(f"early draft warning: {type(exc).__name__}: {exc}", file=sys.stderr)
            return ""

    async def audio_chunks():
        import asyncio
        import numpy as np

        draft_parts: list[Any] = []
        draft_captured = 0
        draft_complete = early_draft_frames <= 0
        while True:
            chunk = await asyncio.to_thread(chunks.get)
            if chunk is None:
                return
            if not draft_complete:
                needed = early_draft_frames - draft_captured
                if needed > 0:
                    part = chunk[:needed]
                    if len(part):
                        draft_parts.append(part.copy())
                        draft_captured += len(part)
                if draft_captured >= early_draft_frames:
                    draft_complete = True
                    draft_audio = np.concatenate(draft_parts)[:early_draft_frames]
                    draft_parts.clear()
                    draft = await asyncio.to_thread(transcribe_draft, draft_audio)
                    if draft and is_current(session_id):
                        emit({
                            "event": "interim",
                            "text": draft,
                            "stable_text": "",
                            "provisional": True,
                            "draft": True,
                        })
            yield chunk

    stream = speech.transcribe(
        audio=audio_chunks(),
        sample_rate=sample_rate,
        timestamps="none",
        stream=True,
    )
    required = max(1, int(os.getenv("PTT_INTERIM_STABILITY", "2")))
    stabilizer = InterimTranscriptStabilizer(required)
    last_raw = ""
    for update in stream:
        if update.get("provisional") is False:
            continue
        raw = normalize_text(str(update.get("text", "")))
        stabilizer.push(raw)
        if not raw or raw == last_raw or not is_current(session_id):
            continue
        last_raw = raw
        stable = stabilizer.visible if raw.startswith(stabilizer.visible) else ""
        emit({
            "event": "interim",
            "text": raw,
            "stable_text": stable,
            "provisional": True,
        })
    result = stream.result()
    return result


def start_client_watchdog() -> None:
    raw_pid = os.getenv("PTT_CLIENT_PID")
    if not raw_pid:
        return
    try:
        client_pid = int(raw_pid)
    except ValueError:
        return
    if client_pid <= 1:
        return

    def watch() -> None:
        interval = threading.Event()
        while not interval.wait(2.0):
            try:
                os.kill(client_pid, 0)
            except OSError:
                # The owning Prime/pi process is gone. Avoid an orphaned model
                # even when its JS shutdown hooks could not run.
                os._exit(0)

    threading.Thread(target=watch, name="ptt-client-watchdog", daemon=True).start()


def main() -> int:
    # Keep stdout clean for the JSON protocol even if dependencies print progress.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import sounddevice as sd

    if "--doctor" in sys.argv or "--doctor-model" in sys.argv:
        requested = os.getenv("PTT_DEVICE", "auto").lower()
        configured_input = os.getenv("PTT_INPUT_DEVICE")
        selected_input: int | str | None = None
        if configured_input:
            selected_input = int(configured_input) if configured_input.isdigit() else configured_input
        errors: list[str] = []
        devices = []
        native_input_rate: int | None = None
        selected_capture_rate: int | None = None
        capture_rate_fallback = False
        try:
            for index, item in enumerate(sd.query_devices()):
                if item["max_input_channels"] > 0:
                    devices.append({"index": index, "name": item["name"], "sample_rate": item["default_samplerate"]})
            input_info = sd.query_devices(selected_input, "input")
            native_input_rate = int(input_info["default_samplerate"])
            selected_capture_rate, capture_rate_fallback = select_capture_sample_rate(
                sd,
                selected_input,
                native_input_rate,
            )
            sd.check_input_settings(
                device=selected_input,
                channels=1,
                dtype="float32",
                samplerate=selected_capture_rate,
            )
        except Exception as exc:
            errors.append(f"microphone: {type(exc).__name__}: {exc}")

        packages: dict[str, str | None] = {}
        for package in ("moondream", "kestrel", "kestrel-kernels", "sounddevice", "torch"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
                errors.append(f"missing package: {package}")

        model_loaded = False
        if "--doctor-model" in sys.argv and not any(error.startswith("missing package:") for error in errors):
            try:
                photon, _speech = load_model(requested, announce=False)
                model_loaded = True
                with contextlib.redirect_stdout(sys.stderr):
                    photon.__exit__(None, None, None)
            except Exception as exc:
                errors.append(f"model: {type(exc).__name__}: {exc}")

        payload = {
            "ok": not errors,
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "device": requested,
            "configured_input": configured_input,
            "input_devices": devices,
            "native_input_rate": native_input_rate,
            "selected_capture_rate": selected_capture_rate,
            "capture_rate_fallback": capture_rate_fallback,
            "packages": packages,
            "model_checked": "--doctor-model" in sys.argv,
            "model_loaded": model_loaded,
            "errors": errors,
        }
        print(json.dumps(payload, indent=2), file=_PROTOCOL_OUT)
        return 0 if not errors else 1

    start_client_watchdog()
    requested = os.getenv("PTT_DEVICE", "auto").lower()
    recorder = Recorder(sd, np)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ptt-transcribe")
    model_future: Future[tuple[Any, Any]] = executor.submit(load_model, requested)

    def report_model_failure(future: Future[tuple[Any, Any]]) -> None:
        try:
            future.result()
        except Exception as exc:
            emit({"event": "model_error", "error": f"{type(exc).__name__}: {exc}"})
            wake_client()

    live_future: Future[dict[str, object]] | None = None
    finalizer_thread: threading.Thread | None = None
    cancel_drain_thread: threading.Thread | None = None
    session_id = 0
    active_session = 0
    session_lock = threading.Lock()
    release_timer: threading.Timer | None = None
    release_generation = 0
    release_lock = threading.Lock()

    def wake_client() -> None:
        client_pid = os.getenv("PTT_CLIENT_PID")
        if client_pid and hasattr(signal, "SIGWINCH"):
            try:
                os.kill(int(client_pid), signal.SIGWINCH)
            except (OSError, ValueError):
                pass

    model_future.add_done_callback(report_model_failure)

    def disarm_release() -> None:
        nonlocal release_timer, release_generation
        with release_lock:
            release_generation += 1
            if release_timer is not None:
                release_timer.cancel()
                release_timer = None

    def arm_release() -> None:
        nonlocal release_timer, release_generation
        with release_lock:
            release_generation += 1
            generation = release_generation
            if release_timer is not None:
                release_timer.cancel()

            def fire() -> None:
                nonlocal release_timer
                with release_lock:
                    if generation != release_generation or release_timer is not timer:
                        return
                    release_timer = None
                emit({"event": "release_timeout"})
                wake_client()

            timer = threading.Timer(0.2, fire)
            release_timer = timer
            timer.daemon = True
            timer.start()

    def is_current(candidate: int) -> bool:
        with session_lock:
            return active_session == candidate

    emit({"event": "ready", "audio": True, "model": "moondream/parakeet-redux", "device": requested})

    try:
        for raw_line in sys.stdin:
            request_id: object | None = None
            try:
                message = json.loads(raw_line)
                if not isinstance(message, dict):
                    raise TypeError("request must be a JSON object")
                request_id = message.get("id")
                if request_id is not None and not isinstance(request_id, (str, int)):
                    raise TypeError("request id must be a string or integer")
                command = message.get("command")
                if not isinstance(command, str):
                    raise TypeError("command must be a string")
                if command == "start":
                    if live_future is not None and not live_future.done():
                        raise RuntimeError("previous transcription is still finishing")
                    if finalizer_thread is not None and finalizer_thread.is_alive():
                        raise RuntimeError("previous transcription is still finalizing")
                    if cancel_drain_thread is not None and cancel_drain_thread.is_alive():
                        raise RuntimeError("cancelled transcription is still draining")
                    chunks: queue.Queue[Any | None] = queue.Queue()
                    recorder.start(chunks)
                    session_id += 1
                    with session_lock:
                        active_session = session_id
                    live_future = executor.submit(
                        transcribe_live,
                        model_future,
                        chunks,
                        recorder.sample_rate,
                        session_id,
                        is_current,
                    )
                    emit({
                        "id": request_id,
                        "ok": True,
                        "event": "recording",
                        "sample_rate": recorder.sample_rate,
                        "native_sample_rate": recorder.native_sample_rate,
                        "capture_rate_fallback": recorder.capture_rate_fallback,
                    })
                elif command == "arm_release":
                    arm_release()
                    emit({"id": request_id, "ok": True, "event": "release_armed"})
                elif command == "disarm_release":
                    disarm_release()
                    emit({"id": request_id, "ok": True, "event": "release_disarmed"})
                elif command == "cancel":
                    disarm_release()
                    with session_lock:
                        active_session = 0
                    recorder.cancel()
                    if live_future is not None:
                        cancelled_future = live_future
                        live_future = None
                        def drain_cancelled() -> None:
                            try:
                                cancelled_future.result(timeout=120)
                            except Exception:
                                pass
                        cancel_drain_thread = threading.Thread(
                            target=drain_cancelled,
                            name="ptt-cancel-drain",
                            daemon=True,
                        )
                        cancel_drain_thread.start()
                    emit({"id": request_id, "ok": True, "event": "cancelled"})
                elif command == "stop":
                    disarm_release()
                    samples, sample_rate = recorder.finish()
                    if live_future is None:
                        raise RuntimeError("live transcription was not started")
                    if finalizer_thread is not None and finalizer_thread.is_alive():
                        raise RuntimeError("transcription is already finalizing")
                    stopped_future = live_future
                    live_future = None
                    stopped_session = session_id
                    stopped_request_id = request_id

                    def finalize_stopped() -> None:
                        try:
                            with contextlib.redirect_stdout(sys.stderr):
                                result = stopped_future.result(timeout=300)
                            if not is_current(stopped_session):
                                emit({"id": stopped_request_id, "ok": False, "error": "transcription cancelled"})
                                return
                            text = normalize_text(str(result.get("text", "")))
                            if len(samples) < sample_rate * 0.12:
                                text = ""
                            emit({"id": stopped_request_id, "ok": True, "event": "transcript", "text": text})
                            wake_client()
                        except Exception as exc:
                            emit({"id": stopped_request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                            wake_client()
                        finally:
                            with session_lock:
                                nonlocal active_session
                                if active_session == stopped_session:
                                    active_session = 0

                    finalizer_thread = threading.Thread(target=finalize_stopped, name="ptt-finalizer", daemon=True)
                    finalizer_thread.start()
                elif command == "devices":
                    devices = []
                    for index, item in enumerate(sd.query_devices()):
                        if item["max_input_channels"] > 0:
                            devices.append({"index": index, "name": item["name"], "sample_rate": item["default_samplerate"]})
                    emit({"id": request_id, "ok": True, "event": "devices", "devices": devices})
                elif command == "shutdown":
                    disarm_release()
                    with session_lock:
                        active_session = 0
                    recorder.cancel()
                    if live_future is not None:
                        shutdown_future = live_future
                        live_future = None
                        def drain_shutdown() -> None:
                            try:
                                shutdown_future.result(timeout=120)
                            except Exception:
                                pass
                        cancel_drain_thread = threading.Thread(
                            target=drain_shutdown,
                            name="ptt-shutdown-drain",
                            daemon=True,
                        )
                        cancel_drain_thread.start()
                    emit({"id": request_id, "ok": True, "event": "shutdown"})
                    break
                else:
                    raise ValueError(f"unknown command: {command!r}")
            except Exception as exc:
                emit({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        disarm_release()
        with session_lock:
            active_session = 0
        try:
            recorder.cancel()
        except Exception as exc:
            print(f"audio cleanup warning: {exc}", file=sys.stderr)
        if live_future is not None:
            try:
                live_future.result(timeout=5.0)
            except TimeoutError:
                os._exit(0)
            except Exception:
                pass
        if cancel_drain_thread is not None and cancel_drain_thread.is_alive():
            cancel_drain_thread.join(timeout=5.0)
            if cancel_drain_thread.is_alive():
                os._exit(0)
        if finalizer_thread is not None and finalizer_thread.is_alive():
            finalizer_thread.join(timeout=5.0)
            if finalizer_thread.is_alive():
                os._exit(0)
        try:
            photon, _speech = model_future.result(timeout=300)
            with contextlib.redirect_stdout(sys.stderr):
                photon.__exit__(None, None, None)
        except Exception as exc:
            print(f"model cleanup warning: {exc}", file=sys.stderr)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
