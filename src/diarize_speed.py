"""
Per-request speed settings for pyannote speaker diarization, shared with the Ultra and ivrit workers
(ronyfadel/runpod-experiments, diarization.py). Nov 25 2026, 22 recordings: TF32 plus one embedding
pass per window cut pyannote's GPU time 55 -> 26 s per audio hour, 99.98% of words keeping their speaker.

  diarize_tf32               true keeps TF32 on for this job (pyannote turns it off on every call)
  diarize_shared_embeddings  true runs the speaker-embedding network once per 10 s window for all its
                             local speakers instead of once per (window, speaker); same embeddings
  diarize_step, diarize_segmentation_batch, diarize_embedding_batch: pyannote's own knobs

A request without them runs exactly as before.
"""
import math
import threading
from contextlib import contextmanager
from typing import Any, Dict

_TF32 = {"on": False}
_SHARED = {"on": False}
_RUN_LOCK = threading.Lock()
_INSTALLED = {"done": False}


def install():
    """Patch pyannote once (after the pipeline is loaded) so the settings can apply per request."""
    if _INSTALLED["done"]:
        return
    _allow_tf32_per_request()
    _allow_shared_embeddings_per_request()
    _INSTALLED["done"] = True


@contextmanager
def applied(pipeline, options: Dict[str, Any]):
    """Apply a request's validated settings to the shared pipeline for one run, then restore them."""
    import torch

    with _RUN_LOCK:
        saved = (pipeline._segmentation.step, pipeline.segmentation_batch_size, pipeline.embedding_batch_size,
                 torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            if "diarize_step" in options:
                pipeline._segmentation.step = options["diarize_step"]
            if "diarize_segmentation_batch" in options:
                pipeline.segmentation_batch_size = options["diarize_segmentation_batch"]
            if "diarize_embedding_batch" in options:
                pipeline.embedding_batch_size = options["diarize_embedding_batch"]
            _TF32["on"] = bool(options.get("diarize_tf32"))
            _SHARED["on"] = bool(options.get("diarize_shared_embeddings"))
            yield
        finally:
            _TF32["on"] = False
            _SHARED["on"] = False
            (pipeline._segmentation.step, pipeline.segmentation_batch_size, pipeline.embedding_batch_size,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = saved


def shared_embeddings(pipeline, file, binary_segmentations, exclude_overlap=False, hook=None):
    """pyannote 4's SpeakerDiarization.get_embeddings, but each window's audio goes through the
    embedding network once with all its local speakers' masks, as (chunks, speakers, frames) weights;
    the WeSpeaker ResNet pools per speaker and returns (chunks, speakers, dimension). pyannote itself
    feeds the same window once per local speaker (3x for community-1), and only the statistics pooling
    differs between those runs. Mask selection (overlap-free mask unless too short) is pyannote's."""
    import numpy as np
    import torch

    raw = binary_segmentations.data
    num_chunks, num_frames, num_speakers = raw.shape
    masks = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    if exclude_overlap:
        embedding = pipeline._embedding
        num_samples = binary_segmentations.sliding_window.duration * embedding.sample_rate
        min_num_frames = math.ceil(num_frames * embedding.min_num_samples / num_samples)
        clean_frames = 1.0 * (np.sum(raw, axis=2, keepdims=True) < 2)
        clean = np.nan_to_num(raw * clean_frames, nan=0.0).astype(np.float32)
        use_clean = np.sum(clean, axis=1) > min_num_frames  # (chunks, speakers)
        masks = np.where(use_clean[:, None, :], clean, masks)

    chunks = [chunk for chunk, _ in binary_segmentations]
    per_batch = max(1, pipeline.embedding_batch_size // max(1, num_speakers))
    model = pipeline._embedding.model_
    device = pipeline._embedding.device
    total = math.ceil(num_chunks / per_batch)
    out = []
    for i, start in enumerate(range(0, num_chunks, per_batch), 1):
        waveforms = torch.vstack([
            pipeline._audio.crop(file, chunk, mode="pad")[0][None] for chunk in chunks[start:start + per_batch]
        ])  # (batch, 1, samples)
        weights = torch.from_numpy(np.ascontiguousarray(masks[start:start + per_batch].transpose(0, 2, 1)))
        with torch.inference_mode():
            batch = model(waveforms.to(device), weights=weights.to(device)).cpu().numpy()
        out.append(batch)  # (batch, speakers, dimension)
        if hook is not None:
            hook("embeddings", batch, total=total, completed=i)
    return np.concatenate(out, axis=0)


def _allow_tf32_per_request():
    """pyannote calls fix_reproducibility (TF32 off, with a warning) at the start of every pipeline
    and inference call. Wrap it where it's imported so a request with diarize_tf32 keeps TF32 on;
    without it, the original runs unchanged."""
    import torch
    import pyannote.audio.core.inference as inference_module
    import pyannote.audio.core.pipeline as pipeline_module

    original = inference_module.fix_reproducibility

    def fix_reproducibility(device):
        if _TF32["on"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return
        original(device)

    inference_module.fix_reproducibility = fix_reproducibility
    pipeline_module.fix_reproducibility = fix_reproducibility


def _allow_shared_embeddings_per_request():
    """Route get_embeddings through shared_embeddings while a request asks for it."""
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    original = SpeakerDiarization.get_embeddings

    def get_embeddings(self, file, binary_segmentations, exclude_overlap=False, hook=None):
        if _SHARED["on"] and not self.training:
            return shared_embeddings(self, file, binary_segmentations, exclude_overlap, hook)
        return original(self, file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook)

    SpeakerDiarization.get_embeddings = get_embeddings


def diarize_options(source: Dict[str, Any]) -> Dict[str, Any]:
    """The speed settings present in a request, validated; unknown or bad values are dropped."""
    options: Dict[str, Any] = {}
    step = source.get("diarize_step")
    if isinstance(step, (int, float)) and 0.1 <= float(step) <= 10.0:
        options["diarize_step"] = float(step)
    for key in ("diarize_segmentation_batch", "diarize_embedding_batch"):
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 512:
            options[key] = value
    for key in ("diarize_tf32", "diarize_shared_embeddings"):
        if isinstance(source.get(key), bool):
            options[key] = source[key]
    return options
