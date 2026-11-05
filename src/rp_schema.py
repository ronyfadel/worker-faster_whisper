INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': False,
        'default': None
    },
    'audio_base64': {
        'type': str,
        'required': False,
        'default': None
    },
    'model': {
        'type': str,
        'required': False,
        'default': 'large-v3'
    },
    'transcription': {
        'type': str,
        'required': False,
        'default': 'plain_text'
    },
    'translate': {
        'type': bool,
        'required': False,
        'default': False
    },
    'translation': {
        'type': str,
        'required': False,
        'default': 'plain_text'
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0
    },
    'best_of': {
        'type': int,
        'required': False,
        'default': 5
    },
    'beam_size': {
        'type': int,
        'required': False,
        'default': 5
    },
    'patience': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'length_penalty': {
        'type': float,
        'required': False,
        'default': 0
    },
    'suppress_tokens': {
        'type': str,
        'required': False,
        'default': '-1'
    },
    'initial_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'condition_on_previous_text': {
        'type': bool,
        'required': False,
        'default': True
    },
    'temperature_increment_on_fallback': {
        'type': float,
        'required': False,
        'default': 0.2
    },
    'compression_ratio_threshold': {
        'type': float,
        'required': False,
        'default': 2.4
    },
    'logprob_threshold': {
        'type': float,
        'required': False,
        'default': -1.0
    },
    'no_speech_threshold': {
        'type': float,
        'required': False,
        'default': 0.6
    },
    'enable_vad': {
        'type': bool,
        'required': False,
        'default': False
    },
    'word_timestamps': {
        'type': bool,
        'required': False,
        'default': False
    },
    'repetition_penalty': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'no_repeat_ngram_size': {
        'type': int,
        'required': False,
        'default': 0
    },
    'suppress_blank': {
        'type': bool,
        'required': False,
        'default': True
    },
    'prefix': {
        'type': str,
        'required': False,
        'default': None
    },
    'hotwords': {
        'type': str,
        'required': False,
        'default': None
    },
    'prompt_reset_on_temperature': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'hallucination_silence_threshold': {
        'type': float,
        'required': False,
        'default': None
    },
    'vad_parameters': {
        'type': dict,
        'required': False,
        'default': None
    },
    # Fallback for audio the speech filter wrongly cuts (distant or echoey
    # speech): when the filter keeps less than this share of the file,
    # transcribe again with vad_fallback_parameters. None turns it off.
    'vad_fallback_min_kept_ratio': {
        'type': float,
        'required': False,
        'default': None,
        'constraints': lambda value: value is None or 0 < value <= 1
    },
    # A None default, not 0.0: the validator skips constraints for values of
    # the default's type, so a float default would let negatives through.
    'vad_fallback_min_duration': {
        'type': float,
        'required': False,
        'default': None,
        'constraints': lambda value: value is None or value >= 0
    },
    'vad_fallback_parameters': {
        'type': dict,
        'required': False,
        'default': None
    },
    'without_timestamps': {
        'type': bool,
        'required': False,
        'default': False
    },
    'max_initial_timestamp': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'prepend_punctuations': {
        'type': str,
        'required': False,
        'default': "\"'“¿([{-"
    },
    'append_punctuations': {
        'type': str,
        'required': False,
        'default': "\"'.。,，!！?？:：”)]}、"
    },
    'multilingual': {
        'type': bool,
        'required': False,
        'default': False
    },
    'language_detection_threshold': {
        'type': float,
        'required': False,
        'default': 0.5
    },
    'language_detection_segments': {
        'type': int,
        'required': False,
        'default': 1
    },
    'clip_timestamps': {
        'type': list,
        'required': False,
        'default': None
    },
    'chunk_length': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_new_tokens': {
        'type': int,
        'required': False,
        'default': None
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 0
    },
    'diarize': {
        'type': bool,
        'required': False,
        'default': False
    },
    'num_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'min_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
}
