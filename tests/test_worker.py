import queue
import unittest

import numpy as np

from ptt_worker import Recorder, normalize_text


class WorkerUtilitiesTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  hello\n  world  "), "hello world")

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


if __name__ == "__main__":
    unittest.main()
