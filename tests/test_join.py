"""Tests for nrl_microstructure.join, on one hand-built match.

Storm (home) v Eels, kicking off at 18:00 Brisbane time (08:00 UTC).
    try 1  Storm, 10 minutes into the game
    try 2  Eels,  30 minutes in
    try 3  Storm, 50 minutes in
"""

import pandas as pd

from nrl_microstructure.join import link_trades, nrl_com_team


MATCHES = pd.DataFrame({
    "match_key": ["nrl_2024_round-1_storm-v-eels"],
    "competition": ["nrl"],
    "home_team": ["Storm"],
    "away_team": ["Eels"],
    "kickoff_utc": ["2024-03-02T08:00:00Z"],
})

TRIES = pd.DataFrame({
    "match_key": ["nrl_2024_round-1_storm-v-eels"] * 3,
    "try_number": [1, 2, 3],
    "game_seconds": [600, 1800, 3000],
    "team": ["Storm", "Eels", "Storm"],
})


def link(dt, team, try_n, result):
    """Link a single trade placed at dt (Brisbane time) and return its row."""
    trades = pd.DataFrame({"dt": [pd.Timestamp(dt)], "comp": ["nrl"], "team": [team],
                           "try_n": [try_n], "result": [result]})
    return link_trades(trades, MATCHES, TRIES).iloc[0]


def test_team_names_are_mapped_to_nrl_com():
    assert nrl_com_team("nrl", "Manly") == ("nrl", "Sea Eagles")
    assert nrl_com_team("nrlw", "Broncos (W)") == ("nrlw", "Broncos")
    assert nrl_com_team("nrl", "NSW (Origin)") == ("origin", "Blues")
    assert nrl_com_team("superleague", "Hull KR (SL)") == (None, None)


def test_a_winning_bet_on_the_team_that_just_scored():
    # Backing the Storm for try 3, placed at 18:55 after the Eels scored try 2.
    row = link("2024-03-02 18:55", "Storm", 3, "won")
    assert row["match_key"] == "nrl_2024_round-1_storm-v-eels"
    assert row["prev_try_team"] == "Eels"
    assert row["next_try_team"] == "Storm"
    assert not row["backed_just_scored"]
    assert row["result_consistent"]
    assert row["timing_consistent"]


def test_a_losing_bet_agrees_with_the_next_try():
    # Backing the Storm for try 2. The Eels scored it, so a loss is consistent.
    row = link("2024-03-02 18:20", "Storm", 2, "lost")
    assert row["backed_just_scored"]
    assert row["result_consistent"]


def test_a_result_that_contradicts_the_match_is_flagged():
    # Recorded as a win, but the Eels scored try 2, not the Storm.
    row = link("2024-03-02 18:20", "Storm", 2, "won")
    assert not row["result_consistent"]


def test_a_bet_placed_before_the_previous_try_is_flagged():
    # A bet on try 3 can't be placed before try 2 (30 minutes in) was scored.
    row = link("2024-03-02 18:15", "Storm", 3, "won")
    assert not row["timing_consistent"]


def test_a_bet_more_than_four_hours_after_kickoff_is_not_linked():
    row = link("2024-03-02 22:30", "Storm", 2, "lost")
    assert row["match_key"] is None
    assert "no nrl match" in row["link_note"]
