import os

from faster_whisper.utils import download_model

model_names = [
    "large-v2",
    "large-v3",
]

DIARIZATION_MODEL = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


def download_diarization_pipeline():
    """
    Download the gated pyannote diarization pipeline. Loading it once pulls
    every file it needs (config plus sub-models) into the Hugging Face cache,
    so the worker can load it offline. Skipped without a token, in which case
    diarization requests fail with an explanatory error.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            f"HF_TOKEN not set; skipping {DIARIZATION_MODEL}. "
            "Diarization will be unavailable in this image."
        )
        return

    from pyannote.audio import Pipeline

    print(f"Downloading {DIARIZATION_MODEL}...")
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not download {DIARIZATION_MODEL}. "
            "Accept its terms on Hugging Face with the account owning HF_TOKEN."
        )
    print(f"Finished downloading {DIARIZATION_MODEL}.")


# Loop through models sequentially
for model_name in model_names:
    download_model_weights(model_name)

download_diarization_pipeline()

print("Finished downloading all models.")
