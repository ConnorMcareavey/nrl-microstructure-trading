"""Two checks against NRL try data from 2013-2025. No fitted model.

1. Lookup table. Take every try in 2013-2019 that was followed by another
   try, and count how often the team that scored also scored the next one.
   Group those moments by the scoring team's pre-match win chance and its
   lead after the try. Then look up each bet in the table. The table's chance
   times the odds taken, minus 1, is the return the history predicts for that
   bet. It uses the odds but not whether the bet won, so the bets' actual
   return is an independent check of it.

2. Rule-change test (the shuffle test). In each match, compare how often the
   same team scored back-to-back with what would be expected if that match's
   tries came in a random order. Its try counts stay the same, so a strong
   team scoring lots of tries isn't mistaken for momentum. Do this for
   2013-2019 and for 2021-2025 (after six-again) and compare.

Run from the repo root, after the scrape and parse:

    python -m nrl_microstructure.history
"""

import numpy as np
import pandas as pd

from nrl_microstructure import clustering, join, odds
from nrl_microstructure.evaluate import load_trades
from nrl_microstructure.parse import PROCESSED_DIR


# Lookup table groups. pd.cut bands include their right edge: (0, 6] means
# a lead of 1 to 6 points.
STRENGTH_BANDS = [0, 0.35, 0.50, 0.65, 0.80, 1.0]
STRENGTH_LABELS = ["under 35%", "35-50%", "50-65%", "65-80%", "over 80%"]
LEAD_BANDS = [-100, 0, 6, 12, 100]
LEAD_LABELS = ["trailing/level", "leads 1-6", "leads 7-12", "leads 13+"]

# Resamples of the historical matches for the range on the predicted return.
N_RESAMPLES = 1000
RANDOM_SEED = 2026


def add_groups(moments):
    """Add the lookup table's two grouping columns."""
    return moments.assign(
        strength=pd.cut(moments["scorer_win_prob"], STRENGTH_BANDS, labels=STRENGTH_LABELS),
        lead=pd.cut(moments["margin_after"], LEAD_BANDS, labels=LEAD_LABELS),
    )


def historical_moments(tries, matches, home_win_prob):
    """Every 2013-2019 try that was followed by another try, grouped, with whether the same team scored next."""
    moments = clustering.post_try_situations(tries, matches, home_win_prob)
    moments = moments[moments["season"].isin(clustering.BEFORE_SIX_AGAIN)
                      & (moments["outcome"] != clustering.NO_FURTHER_TRY)]
    moments = add_groups(moments)
    moments["same_next"] = moments["outcome"] == clustering.SAME_TEAM
    return moments.reset_index(drop=True)


def lookup_table(moments):
    """Share of moments where the same team scored next, and the number of moments, by group."""
    return moments.groupby(["strength", "lead"], observed=True)["same_next"].agg(["mean", "size"])


def core_bets(matches, tries):
    """Link each bet to its match and try; keep NRL Premiership bets on try 2 onwards, not void."""
    linked = join.link_trades(load_trades(), matches, tries)
    return linked[(linked["comp"] == "nrl") & (linked["try_n"] >= 2) & (linked["result"] != "void")
                  & linked["match_key"].fillna("").str.startswith("nrl_")
                  & linked["backed_just_scored"].notna()].copy()


def moment_before_each_bet(bets, tries, home_win_prob):
    """For a bet on try k: the scorer of try k-1, its pre-match win chance and its lead after that try."""
    tries_by_key = tries.set_index(["match_key", "try_number"])
    rows = []
    for index, bet in bets.iterrows():
        prev = tries_by_key.loc[(bet["match_key"], bet["try_n"] - 1)]
        p_home = home_win_prob[bet["match_key"]]
        if prev["is_home"]:
            win_prob, lead = p_home, prev["home_score_after"] - prev["away_score_after"]
        else:
            win_prob, lead = 1 - p_home, prev["away_score_after"] - prev["home_score_after"]
        rows.append({"bet_index": index, "scorer_win_prob": win_prob, "margin_after": lead})
    return add_groups(pd.DataFrame(rows).set_index("bet_index"))


def chance_for_each_bet(table, bets, bet_moments):
    """The table's chance that each bet wins.

    Backing the team that just scored wins if it scores next. Backing the
    other team wins if it doesn't.
    """
    groups = pd.MultiIndex.from_arrays([bet_moments["strength"], bet_moments["lead"]])
    same_next = pd.Series(table["mean"].reindex(groups).to_numpy(), index=bet_moments.index)
    return same_next.where(bets["backed_just_scored"].astype(bool), 1 - same_next)


def predicted_return(chance, bets):
    """Average of chance x odds - 1, the expected return of each bet."""
    return (chance * bets["payout_odds"] - 1).mean()


def predicted_return_resamples(moments, bets, bet_moments, n_resamples=N_RESAMPLES, seed=RANDOM_SEED):
    """Rebuild the table from historical matches drawn with replacement, and recompute the predicted return.

    The spread shows how much the prediction depends on which historical
    matches happened to be in the table.
    """
    rows_by_match = list(moments.groupby("match_key").indices.values())
    rng = np.random.default_rng(seed)
    resamples = np.empty(n_resamples)
    for r in range(n_resamples):
        picked = rng.integers(0, len(rows_by_match), len(rows_by_match))
        resampled = moments.iloc[np.concatenate([rows_by_match[i] for i in picked])]
        chance = chance_for_each_bet(lookup_table(resampled), bets, bet_moments)
        resamples[r] = predicted_return(chance, bets)
    return resamples


def print_lookup(table):
    print("LOOKUP TABLE: 2013-2019 tries followed by another try")
    print("Share where the scoring team also scored the next try (number of moments in brackets)")
    print("Rows: scoring team's pre-match win probability. Columns: its lead after the try.\n")
    cells = table.apply(lambda row: f"{row['mean']:.0%} ({int(row['size'])})", axis=1)
    print(cells.unstack("lead").to_string())
    print()


def print_bets(bets, chance, resamples):
    won = (bets["result"] == "won").mean()
    print(f"TRADES: {len(bets)} NRL Premiership trades, try_n >= 2, voids excluded")
    print("Probability the backed team scores the next try")
    print(f"  implied by the odds (1 / odds)  {bets['implied_prob'].mean():.1%}")
    print(f"  2013-2019 lookup table          {chance.mean():.1%}")
    print(f"  win rate                        {won:.1%}")
    print(f"Lookup table probability above the implied probability on {(chance > bets['implied_prob']).mean():.0%} of trades")
    low, high = np.percentile(resamples, [2.5, 97.5])
    print()
    print(f"PREDICTED ROI: 2013-2019 lookup table, {len(resamples):,} resamples of historical matches, seed {RANDOM_SEED}")
    print(f"  {'':33s}{'ROI':>7s}   {'95% CI':19s}P(ROI <= 0)")
    print(f"  {'predicted from the lookup table':33s}{predicted_return(chance, bets):>+7.1%}   "
          f"{f'{low:+.1%} to {high:+.1%}':19s}{np.mean(resamples <= 0):.3f}")
    print(f"  {'actual flat ROI':33s}{bets['trade_return'].mean():>+7.1%}\n")


def print_rule_change_test(result, per_match):
    old = per_match[per_match["season"].isin(clustering.BEFORE_SIX_AGAIN)]
    new = per_match[per_match["season"].isin(clustering.AFTER_SIX_AGAIN)]
    pct = lambda x: f"{x:+.1%}"
    print("RULE-CHANGE TEST: back-to-back tries before and after six-again")
    print(f"{'':36s}{'2013-2019':>14s}{'2021-2025':>14s}")
    print(f"{'matches':36s}{len(old):>14d}{len(new):>14d}")
    print(f"{'tries per match':36s}{old['n_tries'].mean():>14.2f}{new['n_tries'].mean():>14.2f}")
    same = lambda df: df["observed"].sum() / df["transitions"].sum()
    print(f"{'same team scored next':36s}{same(old):>14.1%}{same(new):>14.1%}")
    print(f"{'excess over random order':36s}{pct(result['before_excess']):>14s}{pct(result['after_excess']):>14s}")
    print(f"{'  95% CI':36s}{pct(result['before_ci'][0]) + ' to ' + pct(result['before_ci'][1]):>14s}"
          f"{'  ' + pct(result['after_ci'][0]) + ' to ' + pct(result['after_ci'][1]):>14s}")
    low, high = result["change_ci"]
    print(f"Change after six-again: {result['change'] * 100:+.1f} points (95% CI {low * 100:+.1f} to {high * 100:+.1f})")
    print(f"Permutation test, 2021-2025 excess > 0: p = {result['after_permutation_p']:.4f}"
          f" ({clustering.N_PERMUTATIONS:,} shuffles)")


def main():
    matches = pd.read_csv(PROCESSED_DIR / "matches.csv")
    tries = pd.read_csv(PROCESSED_DIR / "tries.csv")
    home_win_prob = odds.home_win_probability_from_odds(matches, odds.load_odds())
    nrl = matches[(matches["competition"] == "nrl") & matches["tries_ok"]
                  & matches["match_key"].map(home_win_prob).notna()]

    moments = historical_moments(tries, nrl, home_win_prob)
    table = lookup_table(moments)
    print_lookup(table)

    bets = core_bets(matches, tries)
    bets = bets[bets["match_key"].map(home_win_prob).notna()]
    bet_moments = moment_before_each_bet(bets, tries, home_win_prob)
    chance = chance_for_each_bet(table, bets, bet_moments)
    print_bets(bets, chance, predicted_return_resamples(moments, bets, bet_moments))

    per_match = clustering.per_match_table(tries, matches)
    print_rule_change_test(clustering.compare_before_and_after(per_match), per_match)


if __name__ == "__main__":
    main()
