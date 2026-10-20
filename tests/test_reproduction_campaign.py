"""Check the preregistered selection protocol, not model implementation details."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reproduction_campaign", Path(__file__).resolve().parents[1] / "scripts/run_reproduction_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def scores(coref, temporal, causal, subevent):
    return dict(coreference=coref, temporal=temporal, causal=causal, subevent=subevent,
                overall=(coref + temporal + causal + subevent) / 4)


def test_selection_prefers_balanced_replication_over_one_task_gain():
    reference = scores(90, 55, 30, 27)
    candidates = [("one-task", scores(90, 55, 40, 20)),
                  ("balanced", scores(90.1, 55.1, 30.1, 27.1))]
    winner, ranking = campaign.choose_candidate(reference, candidates)
    assert winner == "balanced"
    assert ranking[0]["minimum_task_delta"] == pytest.approx(0.1)


def test_pairing_rejects_missing_or_different_seed_sets():
    reference = {seed: scores(90, 55, 30, 27) for seed in [13, 21, 42, 87, 100]}
    candidate = dict(reference)
    candidate[200] = candidate.pop(100)
    with pytest.raises(ValueError, match="same five seeds"):
        campaign.summarize_paired(reference, candidate)


def test_paired_uncertainty_uses_seed_matched_differences():
    reference = {seed: scores(80 + i, 50 + i, 25 + i, 20 + i)
                 for i, seed in enumerate([13, 21, 42, 87, 100])}
    candidate = {seed: {key: value + 1 for key, value in row.items()}
                 for seed, row in reference.items()}
    result = campaign.summarize_paired(reference, candidate)
    assert result["causal"]["paired_mean_delta"] == 1
    assert result["causal"]["paired_95pct_ci"] == [1, 1]
    assert result["causal"]["reference_sample_std"] > 0
