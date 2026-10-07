"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import bisect
import gc
import os
import threading
import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.utils import format_timestamp

# Define available models (for validation)
AVAILABLE_MODELS = {
    "large-v2",
    "large-v3",
}

# pyannote speaker diarization pipeline. builder/fetch_models.py bakes it into
# the image when a Hugging Face token is supplied at build time.
DIARIZATION_MODEL = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)
SAMPLE_RATE = 16000


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = (
            threading.Lock()
        )  # Lock for thread-safe model loading/unloading
        self.diarization_pipeline = None
        self.diarization_lock = threading.Lock()

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
        diarize=False,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
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
            # Speakers are assigned per word, so diarization needs word timings.
            "word_timestamps": word_timestamps or diarize,
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

        audio_input = str(audio)
        turns = None
        if diarize:
            # Decode once and share the waveform between Whisper and pyannote.
            audio_input = decode_audio(audio_input, sampling_rate=SAMPLE_RATE)
            turns = self.diarize(
                audio_input,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        segments, info = runner.transcribe(
            audio_input, task="transcribe", **transcribe_kwargs
        )
        segments = list(segments)
        speakers = assign_speakers(segments, turns) if diarize else None

        # Format transcription
        transcription_output = format_segments(transcription, segments, speakers)

        # Handle translation if requested, reusing the transcription settings
        translation_output = None
        if translate:
            translation_segments, _ = runner.transcribe(
                audio_input, task="translate", **transcribe_kwargs
            )
            translation_segments = list(translation_segments)
            translation_output = format_segments(
                translation,
                translation_segments,
                assign_speakers(translation_segments, turns) if diarize else None,
            )

        results = {
            "segments": serialize_segments(segments, speakers),
            "detected_language": info.language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if word_timestamps:
            word_timestamps_list = []
            for i, segment in enumerate(segments):
                for j, word in enumerate(segment.words or []):
                    entry = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }
                    if speakers is not None:
                        entry["speaker"] = speakers[i]["words"][j]
                    word_timestamps_list.append(entry)
            results["word_timestamps"] = word_timestamps_list

        if diarize:
            results["speakers"] = sorted({speaker for _, _, speaker in turns})
            results["diarization"] = [
                {"start": start, "end": end, "speaker": speaker}
                for start, end, speaker in turns
            ]

        return results

    def load_diarization_pipeline(self):
        """
        Lazily load the pyannote pipeline. torch and pyannote are imported
        here so requests that don't diarize never pay for them.
        """
        with self.diarization_lock:
            if self.diarization_pipeline is not None:
                return self.diarization_pipeline

            import torch
            from pyannote.audio import Pipeline

            print(f"Loading diarization pipeline: {DIARIZATION_MODEL}...")
            try:
                pipeline = Pipeline.from_pretrained(
                    DIARIZATION_MODEL, token=os.environ.get("HF_TOKEN")
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load diarization pipeline {DIARIZATION_MODEL}: {e}. "
                    "Diarization is only available when the image was built with a "
                    "Hugging Face token that has access to the model."
                ) from e
            if pipeline is None:
                raise ValueError(
                    f"Diarization pipeline {DIARIZATION_MODEL} is not available. "
                    "Build the image with a Hugging Face token that has accepted "
                    "the model's terms."
                )
            if rp_cuda.is_available():
                pipeline.to(torch.device("cuda"))
            self.diarization_pipeline = pipeline
            print("Diarization pipeline loaded.")
            return pipeline

    def diarize(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
        """
        Run speaker diarization on a 16 kHz mono waveform and return a sorted
        list of (start, end, speaker) turns.
        """
        import torch

        pipeline = self.load_diarization_pipeline()
        speaker_kwargs = {
            key: value
            for key, value in (
                ("num_speakers", num_speakers),
                ("min_speakers", min_speakers),
                ("max_speakers", max_speakers),
            )
            if value is not None
        }
        output = pipeline(
            {
                "waveform": torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": SAMPLE_RATE,
            },
            **speaker_kwargs,
        )
        # pyannote 4 returns an object holding both the regular and the
        # "exclusive" (one speaker at a time) annotations; the exclusive one
        # lines up best with Whisper's single-speaker words. pyannote 3
        # returns the annotation directly.
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        return sorted(
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        )


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


class SpeakerTimeline:
    """
    Looks up which diarization speaker talks the most during a time span.
    Turns are sorted by start; a running max of turn ends lets each lookup
    stop scanning backwards as soon as no earlier turn can still overlap.
    """

    def __init__(self, turns):
        self.turns = turns
        self.starts = [start for start, _, _ in turns]
        self.max_ends = []
        running = float("-inf")
        for _, end, _ in turns:
            running = max(running, end)
            self.max_ends.append(running)

    def speaker_for(self, start, end):
        """
        Return the speaker overlapping [start, end] the most, or the nearest
        speaker when nothing overlaps (e.g. a word in a short pause).
        """
        if not self.turns:
            return None
        idx = bisect.bisect_left(self.starts, end)
        overlap = {}
        j = idx - 1
        while j >= 0 and self.max_ends[j] > start:
            turn_start, turn_end, speaker = self.turns[j]
            shared = min(end, turn_end) - max(start, turn_start)
            if shared > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
            j -= 1
        if overlap:
            return max(overlap, key=overlap.get)

        nearest, best_gap = None, float("inf")
        for k in (idx - 1, idx):
            if 0 <= k < len(self.turns):
                turn_start, turn_end, speaker = self.turns[k]
                gap = max(turn_start - end, start - turn_end, 0.0)
                if gap < best_gap:
                    nearest, best_gap = speaker, gap
        return nearest


def assign_speakers(segments, turns):
    """
    Assign a speaker to every word and segment. Each word gets the speaker it
    overlaps most; a segment gets the speaker holding most of its word time,
    falling back to the segment's own span when it has no words.

    Returns one {"speaker": str, "words": [str, ...]} entry per segment.
    """
    timeline = SpeakerTimeline(turns)
    assigned = []
    for segment in segments:
        word_speakers = []
        talk_time = {}
        for word in segment.words or []:
            speaker = timeline.speaker_for(word.start, word.end)
            word_speakers.append(speaker)
            if speaker is not None:
                talk_time[speaker] = talk_time.get(speaker, 0.0) + max(
                    word.end - word.start, 1e-3
                )
        if talk_time:
            segment_speaker = max(talk_time, key=talk_time.get)
        else:
            segment_speaker = timeline.speaker_for(segment.start, segment.end)
        assigned.append({"speaker": segment_speaker, "words": word_speakers})
    return assigned


def serialize_segments(transcript, speakers=None):
    """
    Serialize the segments to be returned in the API response.
    """
    serialized = []
    for i, segment in enumerate(transcript):
        entry = {
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
        if speakers is not None:
            entry["speaker"] = speakers[i]["speaker"]
        serialized.append(entry)
    return serialized


def segment_speaker(speakers, i):
    """Speaker label for segment i, or None without diarization."""
    if speakers is None:
        return None
    return speakers[i]["speaker"]


def format_segments(format_type, segments, speakers=None):
    """
    Format the segments to the desired format. With diarization,
    formatted_text prefixes each line with its speaker, SRT prefixes each cue
    with the speaker in brackets, and VTT uses <v> voice tags. plain_text
    stays unlabeled.
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        lines = []
        for i, segment in enumerate(segments):
            text = segment.text.lstrip()
            speaker = segment_speaker(speakers, i)
            lines.append(f"{speaker}: {text}" if speaker else text)
        return "\n".join(lines)
    elif format_type == "srt":
        return write_srt(segments, speakers)
    elif format_type == "vtt":
        return write_vtt(segments, speakers)
    else:  # Default or unknown format
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript, speakers=None):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for i, segment in enumerate(transcript):
        text = segment.text.strip().replace('-->', '->')
        speaker = segment_speaker(speakers, i)
        if speaker:
            text = f"<v {speaker}>{text}"
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{text}\n"
        result += "\n"

    return result


def write_srt(transcript, speakers=None):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        text = segment.text.strip().replace('-->', '->')
        speaker = segment_speaker(speakers, i - 1)
        if speaker:
            text = f"[{speaker}] {text}"
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{text}\n"
        result += "\n"

    return result
