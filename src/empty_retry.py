"""
Retry an empty transcript with another model.

large-v2 sometimes returns no segments at all on real speech: 0.6% of speech-heavy production
jobs (Nov 2026), e.g. a 51-s Thai clip (empty on every v2 run) and a 31-s Russian clip (empty
once, fine on reruns). When the caller names `empty_result_fallback_model` and the speech filter
kept enough of the file to say someone is talking, the worker transcribes again with that model in
the same job.
"""


def should_retry_empty(segments, vad_report, model_name, fallback_model, min_kept_ratio):
    """True when the transcript is empty although the speech filter heard speech."""
    if segments or not fallback_model or fallback_model == model_name or not vad_report:
        return False
    # After a filter-off fallback pass, what the filter kept the first time is the speech signal.
    kept = vad_report.get("kept_ratio")
    return kept is not None and kept >= (min_kept_ratio if min_kept_ratio is not None else 0.5)
