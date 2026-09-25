"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import gc
import threading
import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.utils import format_timestamp

# Define available models (for validation)
AVAILABLE_MODELS = {
    "large-v2",
    "large-v3",
}


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = (
            threading.Lock()
        )  # Lock for thread-safe model loading/unloading

    def setup(self):
        """No models are pre-loaded. Setup is minimal."""
        pass

    def predict(
        self,
        audio,
        model_name="large-v3",
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        suppress_tokens="-1",
        suppress_blank=True,
        initial_prompt=None,
        prefix=None,
        hotwords=None,
        condition_on_previous_text=True,
        prompt_reset_on_temperature=0.5,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        hallucination_silence_threshold=None,
        enable_vad=False,
        vad_parameters=None,
        word_timestamps=False,
        without_timestamps=False,
        max_initial_timestamp=1.0,
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:：”)]}、",
        multilingual=False,
        language_detection_threshold=0.5,
        language_detection_segments=1,
        clip_timestamps=None,
        chunk_length=None,
        max_new_tokens=None,
        batch_size=0,
    ):
        """
        Run a single prediction on the model, loading/unloading models as needed.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        with self.model_lock:
            model = None
            if model_name not in self.models:
                # Unload existing model if necessary
                if self.models:
                    existing_model_name = list(self.models.keys())[0]
                    print(f"Unloading model: {existing_model_name}...")
                    # Remove reference and clear dict
                    del self.models[existing_model_name]
                    self.models.clear()
                    # Hint Python to release memory
                    gc.collect()
                    print(f"Model {existing_model_name} unloaded.")

                # Load the requested model
                print(f"Loading model: {model_name}...")
                try:
                    loaded_model = WhisperModel(
                        model_name,
                        device="cuda" if rp_cuda.is_available() else "cpu",
                        compute_type="float16" if rp_cuda.is_available() else "int8",
                    )
                    self.models[model_name] = loaded_model
                    model = loaded_model
                    print(f"Model {model_name} loaded successfully.")
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    raise ValueError(f"Failed to load model {model_name}: {e}") from e
            else:
                # Model already loaded
                model = self.models[model_name]
                print(f"Using already loaded model: {model_name}")

            # Ensure model is loaded before proceeding
            if model is None:
                raise RuntimeError(
                    f"Model {model_name} could not be loaded or retrieved."
                )

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        # Options shared by the transcription and translation passes.
        transcribe_kwargs = {
            "language": language,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "prompt_reset_on_temperature": prompt_reset_on_temperature,
            "initial_prompt": initial_prompt,
            "prefix": prefix,
            "suppress_blank": suppress_blank,
            "suppress_tokens": parse_suppress_tokens(suppress_tokens),
            "without_timestamps": without_timestamps,
            "max_initial_timestamp": max_initial_timestamp,
            "word_timestamps": word_timestamps,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations,
            "multilingual": multilingual,
            "vad_filter": enable_vad,
            "vad_parameters": vad_parameters,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "hallucination_silence_threshold": hallucination_silence_threshold,
            "hotwords": hotwords,
            "language_detection_threshold": language_detection_threshold,
            "language_detection_segments": language_detection_segments,
        }

        if batch_size and batch_size > 0:
            # Batched inference decodes VAD-split chunks in parallel. It only
            # accepts clip_timestamps as a list of {"start", "end"} dicts and
            # always decodes without conditioning on previous text.
            runner = BatchedInferencePipeline(model)
            transcribe_kwargs["batch_size"] = batch_size
            transcribe_kwargs["clip_timestamps"] = to_clip_dicts(clip_timestamps)
        else:
            runner = model
            # The sequential API takes a flat list of start/end seconds or a
            # comma-separated string; "0" means "the whole file".
            transcribe_kwargs["clip_timestamps"] = (
                clip_timestamps if clip_timestamps else "0"
            )

        segments, info = runner.transcribe(
            str(audio), task="transcribe", **transcribe_kwargs
        )
        segments = list(segments)

        # Format transcription
        transcription_output = format_segments(transcription, segments)

        # Handle translation if requested, reusing the transcription settings
        translation_output = None
        if translate:
            translation_segments, _ = runner.transcribe(
                str(audio), task="translate", **transcribe_kwargs
            )
            translation_output = format_segments(
                translation, list(translation_segments)
            )

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if word_timestamps:
            word_timestamps_list = []
            for segment in segments:
                for word in segment.words or []:
                    word_timestamps_list.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                    )
            results["word_timestamps"] = word_timestamps_list

        return results


def parse_suppress_tokens(value):
    """
    Turn the API's comma-separated token id string (e.g. "-1" or "1,2,3") into
    the list of ints faster-whisper expects. "-1" keeps Whisper's default set of
    suppressed special tokens; an empty string disables suppression entirely.
    """
    if value is None:
        return [-1]
    if isinstance(value, (list, tuple)):
        return [int(t) for t in value]
    value = str(value).strip()
    if not value:
        return []
    return [int(t) for t in value.split(",") if t.strip()]


def to_clip_dicts(clip_timestamps):
    """
    Convert a flat [start, end, start, end, ...] list of seconds into the list
    of {"start", "end"} dicts the batched pipeline expects. A trailing start
    with no end runs to the end of the audio.
    """
    if not clip_timestamps:
        return None
    values = [float(v) for v in clip_timestamps]
    clips = []
    for i in range(0, len(values), 2):
        start = values[i]
        end = values[i + 1] if i + 1 < len(values) else float("inf")
        clips.append({"start": start, "end": end})
    return clips


def serialize_segments(transcript):
    """
    Serialize the segments to be returned in the API response.
    """
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in transcript
    ]


def format_segments(format_type, segments):
    """
    Format the segments to the desired format
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":
        return write_vtt(segments)
    else:  # Default or unknown format
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for segment in transcript:
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
