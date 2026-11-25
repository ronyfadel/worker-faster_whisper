"""Needs real numpy and torch (no GPU, no pyannote):
uv run --with numpy --with torch python -m unittest tests/test_shared_embeddings.py

Checks shared_embeddings against pyannote 4.0's own get_embeddings loop (copied below) on a fake
WeSpeaker-like model: same mask choice per (window, speaker), same embeddings."""
import math
import os
import sys
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from diarize_speed import shared_embeddings  # noqa: E402

SAMPLES, FRAMES, SPEAKERS, CHUNKS = 160, 20, 3, 11


class FakeModel:
    """Per-frame features from the waveform, then per-speaker weighted mean and std, like the
    WeSpeaker ResNet's statistics pooling: weights (batch, frames) or (batch, speakers, frames)."""

    def __call__(self, waveforms, weights):
        frames = waveforms.reshape(waveforms.shape[0], FRAMES, -1)  # (b, F, samples per frame)
        feats = torch.stack([frames.mean(-1), frames.abs().amax(-1)], dim=1)  # (b, 2, F)
        w = weights if weights.dim() == 3 else weights[:, None, :]
        total = w.sum(-1, keepdim=True).clamp_min(1e-8)
        mean = torch.einsum("bsf,bdf->bsd", w, feats) / total
        var = torch.einsum("bsf,bsdf->bsd", w, (feats[:, None] - mean[..., None]) ** 2) / total
        out = torch.cat([mean, var.sqrt()], dim=-1)
        return out if weights.dim() == 3 else out[:, 0]


class FakeEmbedding:
    sample_rate = 16000
    min_num_samples = 16000 * 0.004 * 3  # 3 frames' worth at 10 s / 20 frames -> min_num_frames 1
    device = torch.device("cpu")

    def __init__(self):
        self.model_ = FakeModel()

    def __call__(self, waveforms, masks):
        with torch.inference_mode():
            return self.model_(waveforms, weights=masks).numpy()


class FakeSegmentations:
    def __init__(self, data):
        self.data = data
        self.sliding_window = types.SimpleNamespace(duration=0.01)  # 160 samples at 16 kHz

    def __iter__(self):
        for c in range(self.data.shape[0]):
            yield (c, self.data[c])


class FakePipeline:
    def __init__(self, audio):
        self._embedding = FakeEmbedding()
        self.embedding_batch_size = 7
        self.training = False
        self._audio = types.SimpleNamespace(crop=lambda file, chunk, mode: (torch.from_numpy(audio[chunk])[None], 16000))


def reference(pipeline, file, binary_segmentations, exclude_overlap):
    """pyannote 4.0 SpeakerDiarization.get_embeddings, minus the training cache and hook."""
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape
    if exclude_overlap:
        min_num_samples = pipeline._embedding.min_num_samples
        num_samples = duration * pipeline._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
        clean_frames = 1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
        clean = FakeSegmentations(binary_segmentations.data * clean_frames)
    else:
        min_num_frames = -1
        clean = FakeSegmentations(binary_segmentations.data)
    items = []
    for (chunk, masks), (_, clean_masks) in zip(binary_segmentations, clean):
        waveform, _ = pipeline._audio.crop(file, chunk, mode="pad")
        masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
        clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
        for mask, clean_mask in zip(masks.T, clean_masks.T):
            used = clean_mask if np.sum(clean_mask) > min_num_frames else mask
            items.append((waveform[None], torch.from_numpy(used)[None]))
    out = []
    for k in range(0, len(items), pipeline.embedding_batch_size):
        batch = items[k:k + pipeline.embedding_batch_size]
        out.append(pipeline._embedding(torch.vstack([w for w, _ in batch]), torch.vstack([m for _, m in batch])))
    return np.vstack(out).reshape(num_chunks, num_speakers, -1)


class SharedEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.audio = rng.standard_normal((CHUNKS, SAMPLES)).astype(np.float32)
        data = (rng.random((CHUNKS, FRAMES, SPEAKERS)) < 0.4).astype(np.float64)
        data[2, :, :] = 1.0          # all overlap: overlap-free masks empty -> falls back to full masks
        data[3, :, 1] = 0.0          # an inactive local speaker
        data[4, :5, 0] = np.nan      # partial stitching NaNs
        data[5, :, :] = 0.0
        data[5, 0, 2] = 1.0          # a single overlap-free frame: below min_num_frames? (min is 1 -> not >1)
        self.data = data

    def check(self, exclude_overlap):
        pipeline = FakePipeline(self.audio)
        seg = FakeSegmentations(self.data)
        expected = reference(pipeline, None, seg, exclude_overlap)
        got = shared_embeddings(pipeline, None, seg, exclude_overlap)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_matches_pyannote_with_overlap_exclusion(self):
        self.check(True)

    def test_matches_pyannote_without_overlap_exclusion(self):
        self.check(False)


if __name__ == "__main__":
    unittest.main()
