"""Tests for nrl_microstructure.reopening, on small hand-built price histories."""

import numpy as np
import pandas as pd
import pytest

from nrl_microstructure.reopening import reopenings_for_game, scorer_probability_path, traded_clv

T0 = pd.Timestamp("2026-10-15T09:30:00Z")


def point(seconds, try_number, team, odds, suspended=False):
    return {"ts": T0 + pd.Timedelta(seconds=seconds), "try_number": try_number, "team": team,
            "odds_dec": odds, "suspended": suspended}


def test_margin_is_removed_from_the_scorer_probability():
    # 1/1.80 + 1/2.00 = 0.5556 + 0.5 = 1.0556, so p = 0.5556 / 1.0556 = 0.5263.
    market = pd.DataFrame([point(0, 2, "Storm", 1.80), point(0, 2, "Eels", 2.00)])
    path = scorer_probability_path(market, "Storm", "Eels")
    assert path["p"].iloc[-1] == pytest.approx(0.5263, abs=1e-4)


def make_game(storm_odds_after_70s=1.90, suspend_at=None):
    """The Storm score try 1. The 2nd Try market opens with the Storm at 2.20, and moves at 70 seconds."""
    points = [point(0, 2, "Storm", 2.20), point(0, 2, "Eels", 1.70),
              point(70, 2, "Storm", storm_odds_after_70s), point(70, 2, "Eels", 1.95),
              point(600, 3, "Storm", 2.00), point(600, 3, "Eels", 1.80)]
    if suspend_at is not None:
        points += [point(suspend_at, 2, "Storm", 2.20, suspended=True), point(suspend_at, 2, "Eels", 1.70, suspended=True)]
    tries = pd.DataFrame({"try_number": [1, 2], "team": ["Storm", "Eels"]})
    return pd.DataFrame(points).sort_values("ts", kind="stable").reset_index(drop=True), tries


def test_change_after_reopening_is_measured_from_the_opening_price():
    points, tries = make_game()
    rows, _ = reopenings_for_game(points, tries)
    row = rows.iloc[0]
    assert row["scorer"] == "Storm"
    assert row["secs_until_first_change"] == 70
    # Unchanged at 30 and 60 seconds, shorter by 120 seconds.
    assert row["change_30s"] == 0
    assert row["change_60s"] == 0
    assert row["change_120s"] > 0


def test_a_window_with_a_suspension_is_dropped():
    points, tries = make_game(suspend_at=45)
    rows, _ = reopenings_for_game(points, tries)
    row = rows.iloc[0]
    assert row["change_30s"] == 0
    assert np.isnan(row["change_60s"])
    assert np.isnan(row["change_120s"])


def test_traded_clv_compares_the_odds_taken_with_the_later_open_price():
    # Took 2.20 at 10 seconds. The open price was 2.20 at 70 seconds and 1.90 at 130.
    traded = [{"game": "Storm v Eels", "try_number": 2, "team": "Storm", "odds_taken": 2.20, "secs_to_trade": 10,
               "end_secs": 300, "seconds": [0, 100], "odds": [2.20, 1.90], "suspended": [False, False]}]
    clv = traded_clv(traded).iloc[0]
    assert clv["clv_60s"] == pytest.approx(0.0)
    assert clv["clv_120s"] == pytest.approx(2.20 / 1.90 - 1)
