"""
A worker whose GPU ran out of memory gets replaced.

Nov 21 2026: four jobs failed with "CUDA failed with error out of memory" on four workers,
one of them a 60-s clip that failed 2.5 s in. The memory was already taken by earlier jobs
on that worker, so the next job there would fail too. The job still fails (the app retries
it once), but RunPod stops this worker afterwards and starts a fresh one.
"""
import json
import os
import traceback


def is_cuda_out_of_memory(error: BaseException) -> bool:
    '''CTranslate2 raises RuntimeError("CUDA failed with error out of memory"),
    torch raises torch.cuda.OutOfMemoryError ("CUDA out of memory. Tried to allocate …").'''
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


def out_of_memory_result(error: BaseException, runpod_version: str) -> dict:
    '''The failed job as the SDK reports a raised exception (the same JSON error, so the
    backend and the failure log read it unchanged), plus refresh_worker so RunPod stops
    this worker after the job.'''
    return {
        "error": json.dumps({
            "error_type": str(type(error)),
            "error_message": str(error),
            "error_traceback": traceback.format_exc(),
            "hostname": os.environ.get("RUNPOD_POD_HOSTNAME", "unknown"),
            "worker_id": os.environ.get("RUNPOD_POD_ID", "unknown"),
            "runpod_version": runpod_version,
        }),
        "refresh_worker": True,
    }
