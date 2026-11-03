"""Do teams score back-to-back tries more than chance, and did that change after six-again?

The shuffle test counts, in each match, how often the same team scored two
tries in a row, and compares that with what would be expected if the match's
tries had come in a random order. Each team's try count stays the same, so a strong
team scoring lots of tries isn't mistaken for momentum.

The same comparison is also broken down by the game time between consecutive
tries, which shows where the back-to-back tries come from.

Only matches that pass parse.py's try-count check (tries_ok) are used. 2020
is left out: six-again arrived partway through it.
"""

import numpy as np
import pandas as pd


BEFORE_SIX_AGAIN = list(range(2013, 2020))   # 2013-2019
AFTER_SIX_AGAIN = list(range(2021, 2026))    # 2021-2025

RANDOM_SEED = 2026
N_BOOTSTRAP = 20_000
N_PERMUTATIONS = 10_000

# What happened after a try, from the point of view of the team that scored it.
SAME_TEAM, OTHER_TEAM, NO_FURTHER_TRY = 0, 1, 2


# ---------------------------------------------------------------------------
# The shuffle test
# ---------------------------------------------------------------------------

def same_team_transitions(sequence):
    """Count consecutive pairs of tries scored by the same team.

    'AAAABBBA' has 7 transitions (pairs), and 5 of them are same-team:
    AA, AA, AA, BB, BB.
    """
    sequence = list(sequence)
    return sum(1 for previous, current in zip(sequence, sequence[1:]) if previous == current)


def expected_same_team_transitions(n_a, n_b):
    """Expected same-team transitions if the order of tries were random.

    Hold the counts fixed (team A scored n_a tries, team B scored n_b) and put
    the tries in a random order. Each of the N - 1 transitions joins two
    different teams with probability 2 * n_a * n_b / (N * (N - 1)), so on
    average 2 * n_a * n_b / N transitions switch teams and the rest are
    same-team:

        expected = (N - 1) - 2 * n_a * n_b / N

    Example: 5 tries to 3 gives 7 - 2*5*3/8 = 3.25.
    """
    n = n_a + n_b
    if n <= 1:
        return 0.0
    return (n - 1) - 2 * n_a * n_b / n


def per_match_table(tries, matches):
    """One row per usable match: observed and expected same-team transitions.

    Usable = NRL men's, passed the try-count check. Matches with no tries are
    kept (they contribute nothing to either count).
    """
    usable = matches[(matches["competition"] == "nrl") & matches["tries_ok"]]
    tries_by_match = tries.sort_values(["match_key", "try_number"]).groupby("match_key")

    rows = []
    for _, match in usable.iterrows():
        if match["match_key"] in tries_by_match.groups:
            sequence = tries_by_match.get_group(match["match_key"])["is_home"].tolist()
        else:
            sequence = []
        n_home = sum(sequence)
        n_away = len(sequence) - n_home
        rows.append({
            "match_key": match["match_key"],
            "season": match["season"],
            "n_tries": len(sequence),
            "transitions": max(len(sequence) - 1, 0),
            "observed": same_team_transitions(sequence),
            "expected": expected_same_team_transitions(n_home, n_away),
            "sequence": sequence,
        })
    return pd.DataFrame(rows)


def excess_clustering(per_match):
    """(observed - expected) / transitions, over all the matches together.

    0 means tries come in an order no different from random. +0.05 means 5%
    of all transitions are same-team beyond what the match's try counts alone
    would produce.
    """
    return (per_match["observed"] - per_match["expected"]).sum() / per_match["transitions"].sum()


def bootstrap_excess(per_match, n_resamples=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Resample whole matches with replacement; return the excess for each resample."""
    observed_minus_expected = (per_match["observed"] - per_match["expected"]).to_numpy()
    transitions = per_match["transitions"].to_numpy()
    n_matches = len(per_match)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_matches, size=(n_resamples, n_matches))
    return observed_minus_expected[picks].sum(axis=1) / transitions[picks].sum(axis=1)


def permutation_test(per_match, n_permutations=N_PERMUTATIONS, seed=RANDOM_SEED):
    """The chance of seeing this much clustering if tries came in a random order.

    Shuffle the order of tries within every match (keeping each team's count)
    n_permutations times, recompute the excess each time, and count how often
    the shuffled matches cluster at least as much as the real ones. This is a
    one-sided p-value.

    Returns (p_value, shuffled_excesses).
    """
    rng = np.random.default_rng(seed)
    total_same_after_shuffle = np.zeros(n_permutations)

    for sequence in per_match["sequence"]:
        if len(sequence) <= 1:
            continue
        sequence = np.array(sequence)
        # Each row is one random ordering of this match's tries.
        random_keys = rng.random((n_permutations, len(sequence)))
        shuffled = sequence[np.argsort(random_keys, axis=1)]
        total_same_after_shuffle += (shuffled[:, 1:] == shuffled[:, :-1]).sum(axis=1)

    shuffled_excess = (
        (total_same_after_shuffle - per_match["expected"].sum()) / per_match["transitions"].sum()
    )
    observed = excess_clustering(per_match)
    p_value = (1 + np.sum(shuffled_excess >= observed)) / (n_permutations + 1)
    return p_value, shuffled_excess


def interval(resamples):
    return np.percentile(resamples, 2.5), np.percentile(resamples, 97.5)


def compare_before_and_after(per_match, n_bootstrap=N_BOOTSTRAP, n_permutations=N_PERMUTATIONS, seed=RANDOM_SEED):
    """The shuffle test for 2013-2019 and 2021-2025, and the change between them."""
    before = per_match[per_match["season"].isin(BEFORE_SIX_AGAIN)]
    after = per_match[per_match["season"].isin(AFTER_SIX_AGAIN)]

    # Separate seeds for the two periods, so their resamples are independent.
    before_resamples = bootstrap_excess(before, n_bootstrap, seed)
    after_resamples = bootstrap_excess(after, n_bootstrap, seed + 1)
    after_p_value, _ = permutation_test(after, n_permutations, seed + 2)

    return {
        "before_excess": excess_clustering(before), "before_ci": interval(before_resamples),
        "after_excess": excess_clustering(after), "after_ci": interval(after_resamples),
        "after_permutation_p": after_p_value,
        "change": excess_clustering(after) - excess_clustering(before),
        "change_ci": interval(after_resamples - before_resamples),
    }


# ---------------------------------------------------------------------------
# Same team next, by time since the previous try
# ---------------------------------------------------------------------------

GAP_SEASONS = list(range(2013, 2026))   # 2013-2025, so the 2026 trading season isn't used
N_GAP_GROUPS = 10
N_GAP_RESAMPLES = 5_000


def consecutive_pairs(tries, matches, seasons=GAP_SEASONS):
    """One row per pair of consecutive tries in a usable match.

    minutes is the game time between the two tries. random_same is the chance
    the pair is by the same team if the match's tries came in a random order,
    which is the expected same-team transitions spread evenly over the N - 1
    pairs. Example: 5 tries to 3 gives 3.25 / 7 = 46.4%.
    """
    usable = matches[(matches["competition"] == "nrl") & matches["tries_ok"] & matches["season"].isin(seasons)]
    usable_tries = tries[tries["match_key"].isin(usable["match_key"])]
    season_of = usable.set_index("match_key")["season"]

    rows = []
    for match_key, group in usable_tries.sort_values(["match_key", "try_number"]).groupby("match_key"):
        home = group["is_home"].to_numpy()
        seconds = group["game_seconds"].to_numpy()
        n_home = int(home.sum())
        random_same = expected_same_team_transitions(n_home, len(home) - n_home) / max(len(home) - 1, 1)
        for k in range(len(home) - 1):
            rows.append({"match_key": match_key, "season": season_of[match_key],
                         "minutes": (seconds[k + 1] - seconds[k]) / 60,
                         "same": bool(home[k + 1] == home[k]), "random_same": random_same})
    return pd.DataFrame(rows)


def same_team_by_gap(pairs, n_groups=N_GAP_GROUPS, n_resamples=N_GAP_RESAMPLES, seed=RANDOM_SEED):
    """Same-team share against random order, in equal-sized groups by time since the previous try.

    One row per group: its minute range, the number of pairs, the share by the
    same team, the share under random order, the difference, and a 95% CI for
    the difference from resampling whole matches.
    """
    pairs = pairs.assign(group=pd.qcut(pairs["minutes"], n_groups, labels=False),
                         difference=pairs["same"] - pairs["random_same"])

    # Sum each match's pairs within each group, so whole matches can be resampled.
    match_index = pairs["match_key"].astype("category").cat.codes.to_numpy()
    n_matches = match_index.max() + 1
    difference = np.zeros((n_matches, n_groups))
    count = np.zeros((n_matches, n_groups))
    np.add.at(difference, (match_index, pairs["group"]), pairs["difference"])
    np.add.at(count, (match_index, pairs["group"]), 1)

    # Each row of weights says how many times each match is drawn in one resample.
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_matches, np.full(n_matches, 1 / n_matches), size=n_resamples)
    resampled = (weights @ difference) / (weights @ count)

    table = pairs.groupby("group").agg(from_minutes=("minutes", "min"), to_minutes=("minutes", "max"),
                                       pairs=("same", "size"), same=("same", "mean"),
                                       random_same=("random_same", "mean"))
    table["difference"] = table["same"] - table["random_same"]
    table["ci_low"], table["ci_high"] = np.percentile(resampled, [2.5, 97.5], axis=0)
    return table


# ---------------------------------------------------------------------------
# Moments after a try (used by the lookup table in history.py)
# ---------------------------------------------------------------------------

def post_try_situations(tries, matches, home_win_prob):
    """One row per try in a usable NRL match: the situation straight after it.

    home_win_prob is a Series of pre-match home win probabilities, indexed by
    match_key. outcome is what happened next: SAME_TEAM, OTHER_TEAM, or
    NO_FURTHER_TRY.
    """
    usable = matches[(matches["competition"] == "nrl") & matches["tries_ok"]]
    usable_tries = tries[tries["match_key"].isin(usable["match_key"])]
    season_of = usable.set_index("match_key")["season"]

    rows = []
    for match_key, group in usable_tries.sort_values(["match_key", "try_number"]).groupby("match_key"):
        group = group.reset_index(drop=True)
        p_home = home_win_prob[match_key]
        for i, scored in group.iterrows():
            if i + 1 < len(group):
                next_is_home = group.loc[i + 1, "is_home"]
                outcome = SAME_TEAM if next_is_home == scored["is_home"] else OTHER_TEAM
            else:
                outcome = NO_FURTHER_TRY

            if scored["is_home"]:
                scorer_score, other_score, win_prob = scored["home_score_after"], scored["away_score_after"], p_home
            else:
                scorer_score, other_score, win_prob = scored["away_score_after"], scored["home_score_after"], 1 - p_home

            rows.append({
                "match_key": match_key,
                "season": season_of[match_key],
                "scorer_win_prob": win_prob,
                "margin_after": scorer_score - other_score,
                "outcome": outcome,
            })
    return pd.DataFrame(rows)
