"""python -m unittest discover -s tests   (no GPU, model or audio)"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from empty_retry import should_retry_empty  # noqa: E402

SPEECH = {"audio_duration": 31.0, "kept_duration": 30.5, "kept_ratio": 0.992, "fallback_used": False}


class EmptyRetryTest(unittest.TestCase):
    def test_empty_transcript_on_speech_is_retried(self):
        # Nov 18 2026: 31-s Russian clip, filter kept 99%, large-v2 returned 0 segments
        self.assertTrue(should_retry_empty([], SPEECH, "large-v2", "large-v3-turbo", 0.5))
        self.assertTrue(should_retry_empty([], SPEECH, "large-v2", "large-v3-turbo", None))

    def test_not_retried_when_there_is_text_or_no_fallback(self):
        self.assertFalse(should_retry_empty(["segment"], SPEECH, "large-v2", "large-v3-turbo", 0.5))
        self.assertFalse(should_retry_empty([], SPEECH, "large-v2", None, 0.5))
        self.assertFalse(should_retry_empty([], SPEECH, "large-v3-turbo", "large-v3-turbo", 0.5))

    def test_not_retried_on_silence_music_or_without_the_filter(self):
        for vad in ({"kept_ratio": 0}, {"kept_ratio": 0.3, "fallback_used": True}, None, {}):
            self.assertFalse(should_retry_empty([], vad, "large-v2", "large-v3-turbo", 0.5))

    def test_threshold(self):
        self.assertFalse(should_retry_empty([], {"kept_ratio": 0.6}, "large-v2", "large-v3-turbo", 0.7))
        self.assertTrue(should_retry_empty([], {"kept_ratio": 0.7}, "large-v2", "large-v3-turbo", 0.7))


if __name__ == "__main__":
    unittest.main()
