"""
Work around a crash in faster-whisper 1.2.1's word alignment.

When the aligner returns no alignment for a segment's tokens (seen on music, where
Whisper writes a segment that is only a symbol such as "🎵"), `find_alignment` indexes
an empty `time_indices` with a one-element mask and raises

    IndexError: boolean index did not match indexed array along axis 0;
    size of axis is 0 but size of corresponding boolean index is 1

which fails the whole job (production: Spiik music clips, Nov 15 and Nov 17 2026, on the
Whisper and ivrit workers). The method already skips segments it can't split into
words; this copy also skips a segment with no alignment, so that segment gets no word
timings and every other segment keeps its own.

Only applied to the faster-whisper version it was copied from, so an upgrade can't
silently swap in this old body.
"""

from typing import List

import numpy as np

PATCHED_VERSION = "1.2.1"


def find_alignment(
    self,
    tokenizer,
    text_tokens: List[List[int]],
    encoder_output,
    num_frames: int,
    median_filter_width: int = 7,
) -> List[List[dict]]:
    # faster-whisper 1.2.1 WhisperModel.find_alignment, plus the `not alignments` guard.
    if len(text_tokens) == 0:
        return []

    results = self.model.align(
        encoder_output,
        tokenizer.sot_sequence,
        text_tokens,
        num_frames,
        median_filter_width=median_filter_width,
    )
    return_list = []
    for result, text_token in zip(results, text_tokens):
        text_token_probs = result.text_token_probs
        alignments = result.alignments
        if not alignments:
            # The fix: nothing to place this segment's words on.
            return_list.append([])
            continue
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])

        words, word_tokens = tokenizer.split_to_word_tokens(text_token + [tokenizer.eot])
        if len(word_tokens) <= 1:
            return_list.append([])
            continue
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return_list.append([])
            continue

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] / self.tokens_per_second
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probabilities = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]

        return_list.append(
            [
                dict(word=word, tokens=tokens, start=start, end=end, probability=probability)
                for word, tokens, start, end, probability in zip(
                    words, word_tokens, start_times, end_times, word_probabilities
                )
            ]
        )
    return return_list


def apply():
    """Install the fixed method on WhisperModel. Returns True when applied."""
    import faster_whisper
    from faster_whisper import WhisperModel

    if faster_whisper.__version__ != PATCHED_VERSION:
        print(
            f"alignment_patch: faster-whisper {faster_whisper.__version__} is not "
            f"{PATCHED_VERSION}; leaving find_alignment alone"
        )
        return False
    WhisperModel.find_alignment = find_alignment
    return True
