"""Tests for nrl_microstructure.clustering, on synthetic try sequences only."""

import itertools

import numpy as np
import pandas as pd

from nrl_microstructure.clustering import (
    consecutive_pairs,
    excess_clustering,
    expected_same_team_transitions,
    permutation_test,
    same_team_transitions,
)


def test_worked_example():
    # Team A beats Team B 5 tries to 3, in the order AAAABBBA.
    sequence = list("AAAABBBA")
    assert same_team_transitions(sequence) == 5
    assert expected_same_team_transitions(5, 3) == 3.25


def test_expected_formula_matches_brute_force():
    # Average same-team transitions over every distinct ordering of the tries.
    for n_a, n_b in [(1, 1), (2, 1), (3, 2), (4, 4), (5, 0)]:
        orderings = set(itertools.permutations("A" * n_a + "B" * n_b))
        brute_force = np.mean([same_team_transitions(o) for o in orderings])
        assert np.isclose(brute_force, expected_same_team_transitions(n_a, n_b))


def test_consecutive_pairs_worked_example():
    # A made-up Storm (home) v Eels game, 5 tries to 3 in the order SSEESSSE.
    # Random order gives 3.25 same-team pairs out of 7, so 46.4% for each pair.
    matches = pd.DataFrame({"match_key": ["storm-v-eels"], "competition": ["nrl"], "season": [2024],
                            "tries_ok": [True]})
    order = "SSEESSSE"
    tries = pd.DataFrame({"match_key": "storm-v-eels", "try_number": range(1, 9),
                          "is_home": [c == "S" for c in order],
                          "game_seconds": [300, 480, 1500, 1620, 2400, 2580, 3300, 4500]})
    pairs = consecutive_pairs(tries, matches)
    assert len(pairs) == 7
    assert pairs["same"].tolist() == [True, False, True, False, True, True, False]
    assert pairs["minutes"].iloc[0] == 3
    assert np.isclose(pairs["random_same"].iloc[0], 3.25 / 7)


def make_per_match(sequences):
    rows = []
    for sequence in sequences:
        n_a = sum(sequence)
        rows.append({
            "season": 2024,
            "transitions": max(len(sequence) - 1, 0),
            "observed": same_team_transitions(sequence),
            "expected": expected_same_team_transitions(n_a, len(sequence) - n_a),
            "sequence": list(sequence),
        })
    return pd.DataFrame(rows)


def random_order_matches(rng, n_matches=800):
    """Matches whose tries come in a random order: no clustering at all."""
    sequences = []
    for _ in range(n_matches):
        n_tries = rng.integers(2, 12)
        share_a = rng.uniform(0.2, 0.8)      # strength differs from match to match
        sequences.append(list(rng.random(n_tries) < share_a))
    return sequences


def momentum_matches(rng, n_matches=800, stay_probability=0.65):
    """Matches with momentum: each try is by the previous scorer 65% of the time."""
    sequences = []
    for _ in range(n_matches):
        n_tries = rng.integers(2, 12)
        sequence = [rng.random() < 0.5]
        for _ in range(n_tries - 1):
            stay = rng.random() < stay_probability
            sequence.append(sequence[-1] if stay else not sequence[-1])
        sequences.append(sequence)
    return sequences


def test_no_clustering_gives_excess_near_zero_and_no_significance():
    rng = np.random.default_rng(1)
    per_match = make_per_match(random_order_matches(rng))
    assert abs(excess_clustering(per_match)) < 0.03
    p_value, _ = permutation_test(per_match, n_permutations=2000, seed=1)
    assert p_value > 0.05


def test_momentum_is_detected():
    rng = np.random.default_rng(2)
    per_match = make_per_match(momentum_matches(rng))
    assert excess_clustering(per_match) > 0.05
    p_value, _ = permutation_test(per_match, n_permutations=2000, seed=2)
    assert p_value < 0.01


def test_strength_alone_is_not_mistaken_for_clustering():
    # Very lopsided matches (one team scores 90% of tries) with random order.
    # Raw "same team again" is high, but excess clustering should stay near 0,
    # because the expected count already accounts for each match's counts.
    rng = np.random.default_rng(3)
    sequences = [list(rng.random(rng.integers(2, 12)) < 0.9) for _ in range(800)]
    per_match = make_per_match(sequences)
    raw_same_share = per_match["observed"].sum() / per_match["transitions"].sum()
    assert raw_same_share > 0.7
    assert abs(excess_clustering(per_match)) < 0.03
