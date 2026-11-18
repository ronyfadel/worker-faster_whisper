"""python -m unittest discover -s tests   (no GPU, model or audio)"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_echo import is_prompt_echo, prompt_phrases  # noqa: E402

FR = "Voici une phrase en français : Bonjour, comment ça va ?"
SILENT_RERUN = {"audio_duration": 81.0, "kept_duration": 0, "kept_ratio": 0, "fallback_used": True}


def segs(*texts):
    return [SimpleNamespace(text=" " + t) for t in texts]


class PromptEchoTest(unittest.TestCase):
    def test_silent_file_that_only_repeats_the_prompt(self):
        # Nov 17 2026: two near-silent French Spiik files came back as this, filter kept 0 s
        self.assertTrue(is_prompt_echo(segs("Bonjour, comment ça va ?"), FR, SILENT_RERUN))
        self.assertTrue(is_prompt_echo(segs("Bonjour, comment ça va ?", "Bonjour, comment ça va ?"), FR, SILENT_RERUN))
        self.assertTrue(is_prompt_echo(segs(FR), FR, SILENT_RERUN))

    def test_other_languages_and_prompts_without_a_colon(self):
        self.assertTrue(is_prompt_echo(segs("Hallo, wie geht es dir?"), "Das ist ein Satz auf Deutsch. Hallo, wie geht es dir?", SILENT_RERUN))
        self.assertTrue(is_prompt_echo(segs("今日はどうですか？"), "これは日本語のフレーズです：今日はどうですか？", SILENT_RERUN))
        self.assertTrue(is_prompt_echo(segs("这是一个中文短语。"), "这是一个中文短语。", SILENT_RERUN))

    def test_the_phrase_opening_a_real_conversation_is_kept(self):
        self.assertFalse(is_prompt_echo(segs("Bonjour, comment ça va ?", "Très bien."), FR, SILENT_RERUN))

    def test_kept_when_the_filter_heard_speech(self):
        for vad in (
            {"kept_duration": 4.01, "fallback_used": False},
            {"kept_duration": 2.5, "fallback_used": True},
            {"kept_duration": 0, "fallback_used": False},
            None,
        ):
            self.assertFalse(is_prompt_echo(segs("Bonjour, comment ça va ?"), FR, vad))

    def test_no_prompt_or_no_segments(self):
        self.assertFalse(is_prompt_echo(segs("Bonjour, comment ça va ?"), None, SILENT_RERUN))
        self.assertFalse(is_prompt_echo([], FR, SILENT_RERUN))
        self.assertEqual(prompt_phrases(""), set())


if __name__ == "__main__":
    unittest.main()
