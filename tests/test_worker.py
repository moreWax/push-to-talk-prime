import os
import queue
import unittest
from unittest.mock import patch

import numpy as np

from ptt_worker import InterimTranscriptStabilizer, Recorder, normalize_text, select_capture_sample_rate


class WorkerUtilitiesTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  hello\n  world  "), "hello world")

    def test_capture_prefers_model_native_rate(self):
        class FakeSoundDevice:
            def check_input_settings(self, **settings):
                self.settings = settings

        sound = FakeSoundDevice()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTT_CAPTURE_SAMPLE_RATE", None)
            selected, fallback = select_capture_sample_rate(sound, 3, 48_000)
        self.assertEqual(selected, 16_000)
        self.assertFalse(fallback)
        self.assertEqual(sound.settings["samplerate"], 16_000)

    def test_capture_falls_back_when_16k_is_unsupported(self):
        class FakeSoundDevice:
            def check_input_settings(self, **_settings):
                raise RuntimeError("unsupported")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTT_CAPTURE_SAMPLE_RATE", None)
            selected, fallback = select_capture_sample_rate(FakeSoundDevice(), None, 48_000)
        self.assertEqual(selected, 48_000)
        self.assertTrue(fallback)

    def test_capture_native_override_skips_rate_probe(self):
        class FakeSoundDevice:
            def check_input_settings(self, **_settings):
                raise AssertionError("native override must not probe")

        with patch.dict(os.environ, {"PTT_CAPTURE_SAMPLE_RATE": "native"}):
            selected, fallback = select_capture_sample_rate(FakeSoundDevice(), None, 48_000)
        self.assertEqual(selected, 48_000)
        self.assertFalse(fallback)

    def test_capture_retries_native_rate_when_stream_start_fails(self):
        class FakeStream:
            def __init__(self, fail):
                self.fail = fail
                self.closed = False
                self.started = False
            def start(self):
                if self.fail:
                    raise RuntimeError("16k open failed")
                self.started = True
            def stop(self):
                pass
            def close(self):
                self.closed = True

        class FakeSoundDevice:
            def __init__(self):
                self.rates = []
                self.streams = []
            def query_devices(self, _device, _kind):
                return {"default_samplerate": 48_000}
            def check_input_settings(self, **_settings):
                pass
            def InputStream(self, **settings):
                self.rates.append(settings["samplerate"])
                stream = FakeStream(fail=settings["samplerate"] == 16_000)
                self.streams.append(stream)
                return stream

        sound = FakeSoundDevice()
        recorder = Recorder(sound, np)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTT_CAPTURE_SAMPLE_RATE", None)
            os.environ.pop("PTT_DEMO_AUDIO_FILE", None)
            recorder.start()
        self.assertEqual(sound.rates, [16_000, 48_000])
        self.assertTrue(sound.streams[0].closed)
        self.assertTrue(sound.streams[1].started)
        self.assertEqual(recorder.sample_rate, 48_000)
        self.assertTrue(recorder.capture_rate_fallback)
        recorder.cancel()

    def test_capture_leaves_recorder_idle_when_native_retry_fails(self):
        class BrokenStream:
            def start(self):
                raise RuntimeError("open failed")
            def close(self):
                pass

        class FakeSoundDevice:
            def query_devices(self, _device, _kind):
                return {"default_samplerate": 48_000}
            def check_input_settings(self, **_settings):
                pass
            def InputStream(self, **_settings):
                return BrokenStream()

        recorder = Recorder(FakeSoundDevice(), np)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PTT_CAPTURE_SAMPLE_RATE", None)
            os.environ.pop("PTT_DEMO_AUDIO_FILE", None)
            with self.assertRaisesRegex(RuntimeError, "open failed"):
                recorder.start()
        self.assertIsNone(recorder.stream)
        self.assertEqual(recorder.sample_rate, 48_000)
        self.assertTrue(recorder.capture_rate_fallback)

    def test_finish_cleans_up_when_stop_fails(self):
        class BrokenStream:
            closed = False
            def stop(self):
                raise RuntimeError("stop failed")
            def close(self):
                self.closed = True

        recorder = Recorder(None, np)
        stream = BrokenStream()
        recorder.stream = stream
        recorder.frames = [np.array([0.1], dtype=np.float32)]
        live_queue = queue.Queue()
        recorder.live_queue = live_queue
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            recorder.finish()
        self.assertTrue(stream.closed)
        self.assertIsNone(recorder.stream)
        self.assertIsNone(recorder.live_queue)
        self.assertEqual(recorder.frames, [])
        self.assertEqual(live_queue.get_nowait().shape, (5120,))
        self.assertIsNone(live_queue.get_nowait())

    def test_cancel_cleans_up_when_close_fails(self):
        class BrokenStream:
            stopped = False
            def stop(self):
                self.stopped = True
            def close(self):
                raise RuntimeError("close failed")

        recorder = Recorder(None, np)
        stream = BrokenStream()
        recorder.stream = stream
        recorder.frames = [np.array([0.1], dtype=np.float32)]
        live_queue = queue.Queue()
        recorder.live_queue = live_queue
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            recorder.cancel()
        self.assertTrue(stream.stopped)
        self.assertIsNone(recorder.stream)
        self.assertIsNone(recorder.live_queue)
        self.assertEqual(recorder.frames, [])
        self.assertIsNone(live_queue.get_nowait())

    def test_interim_stabilizer_hides_partial_words(self):
        stabilizer = InterimTranscriptStabilizer(2)
        self.assertIsNone(stabilizer.push("Hello, this is a live streaming trans"))
        self.assertEqual(
            stabilizer.push("Hello, this is a live streaming transcrip"),
            "Hello, this is a live streaming",
        )

    def test_interim_stabilizer_hides_repeated_partial_suffix(self):
        stabilizer = InterimTranscriptStabilizer(2)
        self.assertIsNone(stabilizer.push("Hello streaming trans"))
        self.assertEqual(stabilizer.push("Hello streaming trans"), "Hello streaming")

    def test_interim_stabilizer_emits_repeated_complete_text(self):
        stabilizer = InterimTranscriptStabilizer(2)
        self.assertIsNone(stabilizer.push("Hello,"))
        self.assertEqual(stabilizer.push("Hello,"), "Hello,")
        self.assertIsNone(stabilizer.push("Hello, this"))
        self.assertIsNone(stabilizer.push("Hello, this"))
        self.assertEqual(stabilizer.push("Hello, this is"), "Hello, this")

    def test_interim_stabilizer_revises_to_common_word_boundary(self):
        stabilizer = InterimTranscriptStabilizer(2)
        self.assertIsNone(stabilizer.push("we need foo"))
        self.assertEqual(stabilizer.push("we need bar"), "we need")


if __name__ == "__main__":
    unittest.main()
