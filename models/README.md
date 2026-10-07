# Vendored models

## speaker-diarization-community-1

Speaker diarization pipeline used when a request sets `diarize`.

- Source: https://huggingface.co/pyannote/speaker-diarization-community-1
- Author: pyannoteAI
- License: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)

The files are committed here unchanged so that building the image and running
the worker never contact Hugging Face. The model is gated, so refreshing it
needs a Hugging Face token whose account has accepted the model's terms:

```bash
HF_TOKEN=hf_... hf download pyannote/speaker-diarization-community-1 \
  --local-dir models/speaker-diarization-community-1 \
  --exclude "*.gif" --exclude ".gitattributes"
```
