"""Tests for the predicted return in nrl_microstructure.history, on small hand-built examples."""

import pandas as pd

from nrl_microstructure.history import add_groups, chance_for_each_bet, predicted_return


def test_fair_odds_predict_no_edge():
    # If every bet's chance is exactly 1 / odds, the expected return of each
    # bet is chance x odds - 1 = 0, so the predicted return must be zero.
    bets = pd.DataFrame({"payout_odds": [1.60, 2.20, 3.50]})
    chance = 1 / bets["payout_odds"]
    assert abs(predicted_return(chance, bets)) < 1e-12


def test_predicted_return_on_a_hand_worked_example():
    # A 55% chance at 2.20 expects 0.55 x 2.20 - 1 = +0.21.
    # A 40% chance at 2.00 expects 0.40 x 2.00 - 1 = -0.20.
    # The average of the two is +0.005.
    bets = pd.DataFrame({"payout_odds": [2.20, 2.00]})
    chance = pd.Series([0.55, 0.40])
    assert abs(predicted_return(chance, bets) - 0.005) < 1e-12


def test_backing_the_other_team_uses_the_opposite_chance():
    # Two bets in the same group, where the team that just scored got the
    # next try 60% of the time. Backing that team wins 60% of the time.
    # Backing the other team wins the other 40%.
    moments = add_groups(pd.DataFrame({"scorer_win_prob": [0.55, 0.55], "margin_after": [4, 4]}))
    table = pd.DataFrame({"mean": [0.60], "size": [500]},
                         index=pd.MultiIndex.from_tuples([("50-65%", "leads 1-6")], names=["strength", "lead"]))
    bets = pd.DataFrame({"backed_just_scored": [True, False]})
    chance = chance_for_each_bet(table, bets, moments)
    assert list(chance.round(2)) == [0.60, 0.40]
