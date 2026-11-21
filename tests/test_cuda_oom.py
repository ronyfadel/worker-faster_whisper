"""python -m unittest discover -s tests   (no GPU, model or audio)"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuda_oom import is_cuda_out_of_memory, out_of_memory_result  # noqa: E402


class CudaOutOfMemoryTest(unittest.TestCase):
    def test_recognises_ctranslate2_and_torch_messages(self):
        self.assertTrue(is_cuda_out_of_memory(RuntimeError("CUDA failed with error out of memory")))
        self.assertTrue(is_cuda_out_of_memory(RuntimeError(
            "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 19.70 GiB total capacity)")))

    def test_other_errors_are_left_alone(self):
        self.assertFalse(is_cuda_out_of_memory(RuntimeError("CUDA failed with error unspecified launch failure")))
        self.assertFalse(is_cuda_out_of_memory(MemoryError("out of memory")))  # host RAM, not the GPU
        self.assertFalse(is_cuda_out_of_memory(ValueError("Couldn't download the audio")))

    def test_result_fails_the_job_like_the_sdk_and_refreshes_the_worker(self):
        error = RuntimeError("CUDA failed with error out of memory")
        env = {"RUNPOD_POD_ID": "gqodxnloea04s1", "RUNPOD_POD_HOSTNAME": "gqodxnloea04s1-6441282f"}
        with mock.patch.dict(os.environ, env):
            try:
                raise error
            except RuntimeError as caught:
                result = out_of_memory_result(caught, "1.9.0")

        self.assertIs(result["refresh_worker"], True)
        payload = json.loads(result["error"])
        # what the backend (index.js reads error_message) and the failure log read
        self.assertEqual(payload["error_type"], "<class 'RuntimeError'>")
        self.assertEqual(payload["error_message"], "CUDA failed with error out of memory")
        self.assertEqual(payload["worker_id"], "gqodxnloea04s1")
        self.assertEqual(payload["hostname"], "gqodxnloea04s1-6441282f")
        self.assertEqual(payload["runpod_version"], "1.9.0")
        self.assertIn("Traceback", payload["error_traceback"])
        self.assertEqual(set(result), {"error", "refresh_worker"})


if __name__ == "__main__":
    unittest.main()
