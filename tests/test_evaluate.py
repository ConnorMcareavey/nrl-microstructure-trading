"""Tests for nrl_microstructure.evaluate.

The ROI and whole-day tests use small examples worked out by hand. The final
test checks the bootstrap itself. On synthetic trades with no edge, its 95%
interval should contain zero in about 95% of datasets.
"""

import numpy as np
import pandas as pd

from nrl_microstructure.evaluate import (
    add_derived_columns,
    cluster_bootstrap,
    cluster_by_day,
    flat_roi,
    stake_weighted_roi,
    summarise_bootstrap,
)


def group_numbers(labels):
    """Relabel clusters 0, 1, 2, ... in order of first appearance.

    This lets the tests compare groupings without depending on the label
    values themselves.
    """
    codes, _ = pd.factorize(labels)
    return codes.tolist()


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def test_flat_and_stake_weighted_roi_on_a_hand_worked_example():
    # $100 at 3.00 that won (profit +$200, return +2.00 per dollar) and $300
    # at 2.00 that lost (profit -$300, return -1.00 per dollar).
    #
    # Flat ROI           = (2.00 - 1.00) / 2         = +0.50
    # Stake-weighted ROI = (200 - 300) / (100 + 300)  = -0.25
    #
    # The larger losing stake dominates the stake-weighted figure but counts
    # the same as the winner in the flat one.
    trades = add_derived_columns(pd.DataFrame({
        "stake": [100.0, 300.0],
        "profit": [200.0, -300.0],
        "payout_odds": [3.00, 2.00],
        "result": ["won", "lost"],
    }))

    assert flat_roi(trades) == 0.50
    assert stake_weighted_roi(trades) == -0.25


# ---------------------------------------------------------------------------
# Clustering rules
# ---------------------------------------------------------------------------

def test_whole_day_rule():
    # Four trades on 1 June (two matches) and one on 2 June. The whole-day
    # rule puts the four 1 June trades in one cluster.
    trades = pd.DataFrame({
        "dt": pd.to_datetime([
            "2026-06-01 14:00", "2026-06-01 14:30", "2026-06-01 16:00",
            "2026-06-01 18:15", "2026-06-02 14:00",
        ]),
    })
    assert group_numbers(cluster_by_day(trades)) == [0, 0, 0, 0, 1]


# ---------------------------------------------------------------------------
# Zero-edge calibration
# ---------------------------------------------------------------------------

def make_zero_edge_trades(rng, n_matches=160):
    """Synthetic trades with no edge, where each trade wins with probability 1 / odds.

    At those odds the expected profit on every trade is zero, because
    (1 / odds) * (odds - 1) - (1 - 1 / odds) = 0. The true flat and
    stake-weighted ROIs are therefore both zero.

    The data roughly match the core sample, with about 160 matches of 1 to 8
    trades each, odds between 1.60 and 3.50, and stakes from $100 to $1,000.
    """
    trades_per_match = rng.integers(1, 9, size=n_matches)
    match_id = np.repeat(np.arange(n_matches), trades_per_match)
    n_trades = len(match_id)

    odds = rng.uniform(1.60, 3.50, size=n_trades)
    stake = rng.choice([100.0, 200.0, 400.0, 600.0, 1000.0], size=n_trades)
    won = rng.random(n_trades) < 1 / odds
    profit = np.where(won, stake * (odds - 1), -stake)

    return add_derived_columns(pd.DataFrame({
        "match_id": match_id,
        "stake": stake,
        "profit": profit,
        "payout_odds": odds,
        "result": np.where(won, "won", "lost"),
    }))


def test_zero_edge_interval_contains_zero_about_95_percent_of_the_time():
    # Bootstrap 1,000 zero-edge datasets and count how often the 95% interval
    # contains zero. With 1,000 datasets the coverage has a standard error of
    # about 0.7 percentage points, so a correctly calibrated bootstrap lands
    # between 93% and 97%. An overconfident one, covering say 88%, fails.
    n_datasets = 1000
    rng = np.random.default_rng(12345)

    flat_covers_zero = 0
    stake_weighted_covers_zero = 0
    for dataset_number in range(n_datasets):
        trades = make_zero_edge_trades(rng)
        resampled = cluster_bootstrap(
            trades, trades["match_id"], n_resamples=1000, seed=dataset_number
        )

        flat = summarise_bootstrap(resampled["flat"])
        if flat["ci_low"] <= 0 <= flat["ci_high"]:
            flat_covers_zero += 1

        stake_weighted = summarise_bootstrap(resampled["stake_weighted"])
        if stake_weighted["ci_low"] <= 0 <= stake_weighted["ci_high"]:
            stake_weighted_covers_zero += 1

    flat_coverage = flat_covers_zero / n_datasets
    stake_weighted_coverage = stake_weighted_covers_zero / n_datasets

    assert 0.93 <= flat_coverage <= 0.97, f"flat coverage {flat_coverage:.3f}"
    assert 0.93 <= stake_weighted_coverage <= 0.97, (
        f"stake-weighted coverage {stake_weighted_coverage:.3f}"
    )
