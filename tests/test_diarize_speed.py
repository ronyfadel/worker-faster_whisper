"""python -m unittest tests/test_diarize_speed.py   (no GPU, model, torch or pyannote)"""
import ast
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = types.ModuleType("torch")
torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                                       cudnn=types.SimpleNamespace(allow_tf32=False))
sys.modules.setdefault("torch", torch)

import diarize_speed  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


class FakePipeline:
    def __init__(self):
        self._segmentation = types.SimpleNamespace(step=1.0)
        self.segmentation_batch_size = 32
        self.embedding_batch_size = 32


class DiarizeSpeedTest(unittest.TestCase):
    def test_only_valid_settings_are_kept(self):
        self.assertEqual(diarize_speed.diarize_options({"diarize_tf32": None, "diarize_shared_embeddings": None}), {})
        self.assertEqual(diarize_speed.diarize_options({"diarize_tf32": True, "diarize_shared_embeddings": True}),
                         {"diarize_tf32": True, "diarize_shared_embeddings": True})
        self.assertEqual(diarize_speed.diarize_options({"diarize_tf32": "yes", "diarize_step": 0}), {})

    def test_settings_apply_for_one_run_and_are_restored(self):
        pipeline = FakePipeline()
        with diarize_speed.applied(pipeline, {"diarize_tf32": True, "diarize_shared_embeddings": True,
                                              "diarize_step": 2.0}):
            self.assertTrue(diarize_speed._TF32["on"])
            self.assertTrue(diarize_speed._SHARED["on"])
            self.assertEqual(pipeline._segmentation.step, 2.0)
        self.assertFalse(diarize_speed._TF32["on"])
        self.assertFalse(diarize_speed._SHARED["on"])
        self.assertEqual(pipeline._segmentation.step, 1.0)

    def test_no_settings_change_nothing(self):
        pipeline = FakePipeline()
        with diarize_speed.applied(pipeline, {}):
            self.assertFalse(diarize_speed._TF32["on"])
            self.assertFalse(diarize_speed._SHARED["on"])
            self.assertEqual((pipeline._segmentation.step, pipeline.embedding_batch_size), (1.0, 32))


class PredictPlumbingTest(unittest.TestCase):
    """predict() passes its keywords to _predict() positionally; the two lists must stay in step."""

    def test_predict_and_private_predict_agree(self):
        tree = ast.parse(open(os.path.join(SRC, "predict.py")).read())
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        public = [a.arg for a in funcs["predict"].args.args][1:]  # minus self
        private = [a.arg for a in funcs["_predict"].args.args][1:]
        call = next(n for n in ast.walk(funcs["predict"]) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", "") == "_predict")
        passed = [a.id for a in call.args]
        self.assertEqual(passed, private)
        self.assertIn("diarize_tf32", public)
        self.assertIn("diarize_shared_embeddings", public)
        # every schema key is a predict keyword (the handler passes them 1:1)
        schema_tree = ast.parse(open(os.path.join(SRC, "rp_schema.py")).read())
        keys = [k.value for n in ast.walk(schema_tree) if isinstance(n, ast.Dict)
                for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
                and k.value not in ("type", "required", "default", "constraints")]
        for key in ("diarize_tf32", "diarize_shared_embeddings"):
            self.assertIn(key, keys)
        missing = [k for k in keys if k not in public and k not in ("audio", "audio_base64", "model", "model_name")]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
