"""
Whisper repeating the job's own initial_prompt back as the transcript.

gcp primes every job with a sentence in its language plus an example phrase ("Voici une phrase
en français : Bonjour, comment ça va ?"). When the speech filter kept nothing and the filter-off
fallback pass ran, Whisper can return just that phrase: two near-silent French Spiik files on
Nov 17 2026 came back as "Bonjour, comment ça va ?".

The phrase is also something people say ("Bonjour, comment ça va ? Très bien…"), so the
transcript is only dropped when the filter heard nothing (kept 0 s) AND every segment is the
prompt or its example phrase. Any other text keeps everything.
"""

import re
import unicodedata

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _normalize(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _NON_WORD.sub(" ", text).strip()


def prompt_phrases(initial_prompt):
    """The prompt itself and its example phrase (after the ":" or the first sentence)."""
    if not initial_prompt:
        return set()
    parts = re.split(r"[:：]|(?<=[.。])\s", initial_prompt)
    return {phrase for phrase in (_normalize(initial_prompt), _normalize(parts[-1])) if phrase}


def _is_only_phrases(text, phrases):
    rest = _normalize(text)
    if not rest:
        return False
    ordered = sorted(phrases, key=len, reverse=True)
    while rest:
        match = next((p for p in ordered if rest == p or rest.startswith(p + " ")), None)
        if match is None:
            return False
        rest = rest[len(match):].strip()
    return True


def is_prompt_echo(segments, initial_prompt, vad_report):
    """True when the filter kept 0 s and every segment only repeats the prompt."""
    if not segments or not vad_report or not vad_report.get("fallback_used"):
        return False
    if vad_report.get("kept_duration") != 0:
        return False
    phrases = prompt_phrases(initial_prompt)
    return bool(phrases) and _is_only_phrases(" ".join(s.text for s in segments), phrases)
