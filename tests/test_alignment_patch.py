"""
python -m unittest discover -s tests   (needs faster-whisper==1.2.1 and numpy; no GPU)

Reproduces the production crash (Nov 15/17 2026, Spiik music clip whose filter-off pass
returned a single "🎵" segment) with a stand-in aligner, without any model or audio.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import faster_whisper  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

import alignment_patch  # noqa: E402

EOT = 50257


class FakeTokenizer:
    sot_sequence = (50258,)
    eot = EOT

    def split_to_word_tokens(self, tokens):
        # One "word" per token, then the end-of-text marker, as the real tokenizer does.
        body = [t for t in tokens if t != EOT]
        return [f"w{t}" for t in body] + ["<eot>"], [[t] for t in body] + [[EOT]]


class FakeCT2Model:
    """Stands in for ctranslate2's model.align: one result per segment."""

    def __init__(self, results):
        self.results = results

    def align(self, *args, **kwargs):
        return self.results


def result(alignments, probs):
    return SimpleNamespace(alignments=alignments, text_token_probs=probs)


def fake_whisper(results):
    return SimpleNamespace(model=FakeCT2Model(results), tokens_per_second=50)


class FindAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        # Segment 1: a normal two-word segment. Segment 2: a lone "🎵" token the aligner
        # returned nothing for.
        self.tokens = [[11, 12], [9992]]
        self.results = [
            result([(0, 0), (1, 25), (2, 50)], [0.9, 0.8, 0.7]),
            result([], [0.1]),
        ]

    def test_stock_method_crashes_on_a_segment_without_alignment(self):
        if alignment_patch.upstream_has_fix(WhisperModel.find_alignment):
            # upstream main: the stock method already returns no words for that segment
            words = WhisperModel.find_alignment(fake_whisper(self.results), self.tokenizer, self.tokens, None, 100)
            self.assertEqual(words[1], [])
            return
        with self.assertRaises(IndexError) as raised:
            WhisperModel.find_alignment(
                fake_whisper(self.results), self.tokenizer, self.tokens, None, 100
            )
        self.assertIn("boolean index did not match", str(raised.exception))

    def test_patched_method_skips_only_that_segment(self):
        words = alignment_patch.find_alignment(
            fake_whisper(self.results), self.tokenizer, self.tokens, None, 100
        )
        self.assertEqual(len(words), 2)
        self.assertEqual([w["word"] for w in words[0]], ["w11", "w12"])
        self.assertEqual([(w["start"], w["end"]) for w in words[0]], [(0.0, 0.5), (0.5, 1.0)])
        self.assertEqual(words[1], [])

    def test_patched_method_matches_stock_when_everything_aligns(self):
        aligned = self.results[:1]
        tokens = self.tokens[:1]
        stock = WhisperModel.find_alignment(fake_whisper(aligned), self.tokenizer, tokens, None, 100)
        patched = alignment_patch.find_alignment(
            fake_whisper(aligned), self.tokenizer, tokens, None, 100
        )
        self.assertEqual(stock, patched)

    def test_empty_input(self):
        self.assertEqual(
            alignment_patch.find_alignment(fake_whisper([]), self.tokenizer, [], None, 100), []
        )

    def test_upstream_fix_is_detected(self):
        # our own copy has the guard; stock 1.2.1 doesn't
        self.assertTrue(alignment_patch.upstream_has_fix(alignment_patch.find_alignment))
        if not alignment_patch.upstream_has_fix(WhisperModel.find_alignment):
            self.assertFalse(alignment_patch.upstream_has_fix(WhisperModel.find_alignment))

    def test_apply_installs_only_on_the_copied_version(self):
        original = WhisperModel.find_alignment
        if alignment_patch.upstream_has_fix(original):
            self.assertFalse(alignment_patch.apply())   # upstream main: nothing to patch
            self.assertIs(WhisperModel.find_alignment, original)
            return
        try:
            self.assertEqual(faster_whisper.__version__, alignment_patch.PATCHED_VERSION)
            self.assertTrue(alignment_patch.apply())
            self.assertIs(WhisperModel.find_alignment, alignment_patch.find_alignment)
            WhisperModel.find_alignment = original
            real_version = faster_whisper.__version__
            faster_whisper.__version__ = "9.9.9"
            try:
                self.assertFalse(alignment_patch.apply())
                self.assertIs(WhisperModel.find_alignment, original)
            finally:
                faster_whisper.__version__ = real_version
        finally:
            WhisperModel.find_alignment = original


if __name__ == "__main__":
    unittest.main()
