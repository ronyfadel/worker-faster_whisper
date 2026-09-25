![Faster Whisper Logo](https://5ccaof7hvfzuzf4p.public.blob.vercel-storage.com/banner-pjbGKw0buxbWGhMVC165Gf9qgqWo7I.jpeg)

[Faster Whisper](https://github.com/guillaumekln/faster-whisper) is designed to process audio files using various Whisper models, with options for transcription formatting, language translation and more.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-faster_whisper)](https://www.runpod.io/console/hub/runpod-workers/worker-faster_whisper)

---

## Models

- large-v2
- large-v3

## Input

| Input                               | Type  | Description                                                                                                                                                            |
| ----------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio`                             | Path  | URL to Audio file                                                                                                                                                      |
| `audio_base64`                      | str   | Base64-encoded audio file                                                                                                                                              |
| `model`                             | str   | Choose a Whisper model. Choices: "large-v2", "large-v3". Default: "large-v3"                                                                                           |
| `transcription`                     | str   | Choose the format for the transcription. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                  |
| `translate`                         | bool  | Translate the text to English when set to True. Default: False                                                                                                         |
| `translation`                       | str   | Choose the format for the translation. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                    |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: None                                                                                |
| `temperature`                       | float | Temperature to use for sampling. Default: 0                                                                                                                            |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                                                                               |
| `beam_size`                         | int   | Number of beams in beam search, only applicable when temperature is zero. Default: 5                                                                                   |
| `patience`                          | float | Optional patience value to use in beam decoding. Default: None                                                                                                         |
| `length_penalty`                    | float | Optional token length penalty coefficient (alpha). Default: None                                                                                                       |
| `suppress_tokens`                   | str   | Comma-separated list of token ids to suppress during sampling. Default: "-1"                                                                                           |
| `initial_prompt`                    | str   | Optional text to provide as a prompt for the first window. Default: None                                                                                               |
| `condition_on_previous_text`        | bool  | If True, provide the previous output of the model as a prompt for the next window. Default: True                                                                       |
| `temperature_increment_on_fallback` | float | Temperature to increase when falling back when the decoding fails. Default: 0.2                                                                                        |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is higher than this value, treat the decoding as failed. Default: 2.4                                                                    |
| `logprob_threshold`                 | float | If the average log probability is lower than this value, treat the decoding as failed. Default: -1.0                                                                   |
| `no_speech_threshold`               | float | If the probability of the token is higher than this value, consider the segment as silence. Default: 0.6                                                               |
| `enable_vad`                        | bool  | If True, use the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model. Default: False               |
| `word_timestamps`                   | bool  | If True, include word timestamps in the output. Default: False                                                                                                         |
| `repetition_penalty` | float | Penalty applied to the score of previously generated tokens (>1 discourages repetition). Default: 1.0 |
| `no_repeat_ngram_size` | int | Prevent repetitions of n-grams with this size (0 disables). Default: 0 |
| `suppress_blank` | bool | Suppress blank outputs at the beginning of the sampling. Default: True |
| `prefix` | str | Optional text to provide as a prefix for the first window. Default: None |
| `hotwords` | str | Words or phrases to boost during decoding, e.g. names or jargon. Ignored when `prefix` is set. Default: None |
| `prompt_reset_on_temperature` | float | Reset the prompt if a fallback temperature is above this value. Only used when `condition_on_previous_text` is True. Default: 0.5 |
| `hallucination_silence_threshold` | float | When `word_timestamps` is True, skip silent periods longer than this many seconds when a possible hallucination is detected. Default: None |
| `vad_parameters` | dict | Silero VAD options: `threshold`, `neg_threshold`, `min_speech_duration_ms`, `max_speech_duration_s`, `min_silence_duration_ms`, `speech_pad_ms`. Default: None |
| `without_timestamps` | bool | Only sample text tokens, no timestamp tokens. Default: False |
| `max_initial_timestamp` | float | The initial timestamp cannot be later than this many seconds. Default: 1.0 |
| `prepend_punctuations` | str | When `word_timestamps` is True, merge these punctuation symbols with the next word. Default: "'“¿([{- |
| `append_punctuations` | str | When `word_timestamps` is True, merge these punctuation symbols with the previous word. Default: "'.。,，!！?？:：”)]}、 |
| `multilingual` | bool | Perform language detection on every segment, for code-switched audio. Default: False |
| `language_detection_threshold` | float | Stop language detection once a language's probability exceeds this value. Default: 0.5 |
| `language_detection_segments` | int | Number of 30-second segments to consider for language detection. Default: 1 |
| `clip_timestamps` | list | Flat list of `[start, end, start, end, ...]` seconds to transcribe; a trailing start runs to the end of the file. Default: whole file |
| `chunk_length` | int | Length of audio window in seconds. Default: model's window (30) |
| `max_new_tokens` | int | Maximum number of new tokens to generate per chunk. Default: None |
| `batch_size` | int | When > 0, use faster-whisper's `BatchedInferencePipeline` and decode up to this many VAD-split chunks in parallel. Needs `enable_vad` or `clip_timestamps` for audio over 30 seconds; `condition_on_previous_text` is ignored. Default: 0 (sequential) |

### Example

The following inputs can be used for testing the model:

```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "model": "large-v3"
  }
}
```

producing an output like this:

```json
{
  "segments": [
    {
      "id": 1,
      "seek": 106,
      "start": 0.11,
      "end": 3.11,
      "text": " Hello and welcome!",
      "tokens": [50364, 25, 7, 287, 50514],
      "temperature": 0.1,
      "avg_logprob": -0.8348079785480325,
      "compression_ratio": 0.5789473684210527,
      "no_speech_prob": 0.1453857421875
    }
  ],
  "detected_language": "en",
  "transcription": "Hello and welcome!",
  "translation": null,
  "device": "cuda",
  "model": "large-v3",
  "translation_time": 0.3796223163604736
}
```
