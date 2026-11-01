"""Tests for nrl_microstructure.odds, on small hand-built odds files."""

import pandas as pd
import pytest

from nrl_microstructure.odds import home_win_probability_from_odds, load_odds


def write_odds(tmp_path, rows):
    path = tmp_path / "odds.csv"
    pd.DataFrame(rows, columns=["Date", "Home Team", "Away Team", "Home Odds", "Away Odds"]).to_csv(path, index=False)
    return path


def test_margin_is_removed(tmp_path):
    # 1/1.50 + 1/2.60 = 0.6667 + 0.3846 = 1.0513, so the margin is about 5%.
    # Dividing by the total gives p_home = 0.6667 / 1.0513 = 0.6341.
    path = write_odds(tmp_path, [["2024-03-02", "Penrith Panthers", "Melbourne Storm", 1.50, 2.60]])
    odds = load_odds(path)
    assert odds.loc[0, "p_home"] == pytest.approx(0.6341, abs=1e-4)
    assert (odds.loc[0, "home_team"], odds.loc[0, "away_team"]) == ("Panthers", "Storm")


def test_unknown_team_name_is_an_error(tmp_path):
    path = write_odds(tmp_path, [["2024-03-02", "Penrith Panthers", "Perth Reds", 1.50, 2.60]])
    with pytest.raises(ValueError, match="Perth Reds"):
        load_odds(path)


def make_matches(home, away, kickoff_utc):
    return pd.DataFrame({"match_key": ["m"], "home_team": [home], "away_team": [away], "kickoff_utc": [kickoff_utc]})


def make_odds(date, home, away, p_home):
    return pd.DataFrame({"Date": [pd.Timestamp(date)], "home_team": [home], "away_team": [away], "p_home": [p_home]})


def test_teams_listed_the_other_way_round_are_flipped():
    # nrl.com has the Storm at home; the odds file has the Panthers at home
    # with a 60% chance, so the Storm's chance is 40%.
    matches = make_matches("Storm", "Panthers", "2024-03-02T08:00:00Z")
    odds = make_odds("2024-03-02", "Panthers", "Storm", 0.60)
    assert home_win_probability_from_odds(matches, odds)["m"] == pytest.approx(0.40)


def test_kickoff_date_is_taken_in_sydney_time():
    # 14:00 UTC on 1 March is 1:00 am on 2 March in Sydney, so a 2 March odds
    # row matches even though the UTC date is the 1st.
    matches = make_matches("Panthers", "Storm", "2024-03-01T14:00:00Z")
    odds = make_odds("2024-03-02", "Panthers", "Storm", 0.60)
    assert home_win_probability_from_odds(matches, odds)["m"] == pytest.approx(0.60)


def test_odds_more_than_a_day_away_do_not_match():
    matches = make_matches("Panthers", "Storm", "2024-03-02T08:00:00Z")
    odds = make_odds("2024-03-05", "Panthers", "Storm", 0.60)
    assert pd.isna(home_win_probability_from_odds(matches, odds)["m"])
