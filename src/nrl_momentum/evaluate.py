"""Evaluate the trades in data/trades.csv.

Each row of trades.csv is one bet in a bookmaker's in-play "next try" market.
This module reports, in order, the return on the core sample, a
match-clustered bootstrap interval for it, the same interval with every trade
on the same day treated as linked, a breakdown by competition, and the win
rate against the implied probability by odds band.

Run from the repository root with

    python -m nrl_momentum.evaluate

All random draws use a fixed seed, so the output is the same on every run.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRADES_PATH = REPO_ROOT / "data" / "trades.csv"

# Seed for every bootstrap in this module.
RANDOM_SEED = 2026

# Resamples per bootstrap interval.
N_RESAMPLES = 20_000

COMPETITION_NAMES = {
    "nrl": "NRL",
    "nrlw": "NRLW",
    "international": "World Cup",
    "superleague": "Super League",
}

# trades.csv records State of Origin trades under "nrl". They are reported as
# a separate competition and excluded from the core sample.
ORIGIN_TEAMS = {"NSW (Origin)", "QLD (Origin)"}

# Odds bands for the win-rate table. Each band includes its upper edge, so a
# trade at exactly 1.70 falls in "<= 1.70" and one at 2.00 in "1.70 - 2.00".
ODDS_BAND_EDGES = [1.0, 1.70, 2.00, 2.30, 2.70, np.inf]
ODDS_BAND_LABELS = ["<= 1.70", "1.70 - 2.00", "2.00 - 2.30", "2.30 - 2.70", "> 2.70"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trades(path=DEFAULT_TRADES_PATH):
    """Read trades.csv and add the derived columns used throughout the module."""
    trades = pd.read_csv(path, parse_dates=["dt"])
    return add_derived_columns(trades)


def add_derived_columns(trades):
    """Add the trade_return, implied_prob and won columns.

    trade_return is profit per dollar staked (+1.50 for a win at 2.50, -1.00
    for a loss, 0.00 for a void). Its mean is the flat ROI.

    implied_prob is 1 / payout_odds, the win probability implied by the price,
    including the bookmaker's margin. For winning trades payout_odds is the
    price actually paid (return / stake), which can be slightly above the
    quoted price.

    won is True for winning trades and False for losses, cash-outs and voids.
    """
    trades = trades.copy()
    trades["trade_return"] = trades["profit"] / trades["stake"]
    trades["implied_prob"] = 1 / trades["payout_odds"]
    trades["won"] = trades["result"] == "won"
    return trades


def competition_names(trades):
    """Competition label for each trade, with State of Origin shown separately."""
    names = trades["comp"].map(COMPETITION_NAMES).fillna(trades["comp"])
    return names.where(~trades["team"].isin(ORIGIN_TEAMS), "State of Origin")


def core_sample(trades):
    """NRL Premiership trades on the second try of a match or later, excluding voids.

    Trades on the first try are excluded because no try had been scored when
    they were placed. Voids are excluded because the stake was returned.
    State of Origin, NRLW, World Cup and Super League trades appear only in
    the competition table. This definition was chosen after the season had
    been played.
    """
    is_nrl = (trades["comp"] == "nrl") & ~trades["team"].isin(ORIGIN_TEAMS)
    is_second_try_or_later = trades["try_n"] >= 2
    is_settled = trades["result"] != "void"
    return trades[is_nrl & is_second_try_or_later & is_settled].copy()


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def flat_roi(trades):
    """Mean return per trade, with every trade weighted equally regardless of stake."""
    return trades["trade_return"].mean()


def stake_weighted_roi(trades):
    """Total profit divided by total stake, so larger trades carry more weight."""
    return trades["profit"].sum() / trades["stake"].sum()


# ---------------------------------------------------------------------------
# Clustering rules
#
# Trades in the same match are not independent, because a mispriced match
# affects every trade placed on it. The bootstrap therefore resamples whole
# clusters of trades. Each function below returns one cluster label per
# trade, aligned with the rows of `trades`.
#
# The primary rule groups trades by match. The whole-day rule is a
# conservative check that also treats different matches on the same day as
# linked.
# ---------------------------------------------------------------------------

def cluster_by_match_id(trades):
    """Group trades by the match_id column. This is the primary rule.

    Each match_id is one real match, numbered m1, m2, ... in order of the
    first trade on it. The matches were identified by linking each trade to
    nrl.com's match data (see join.py).
    """
    return trades["match_id"]


def cluster_by_day(trades):
    """Group all trades placed on the same calendar day.

    This is the most conservative rule. It merges matches played on the same
    day, giving the fewest clusters.
    """
    check_timestamps(trades)
    return trades["dt"].dt.date


def check_timestamps(trades):
    """Raise an error if any trade has no timestamp.

    Without this check, trades with missing times would silently drop out of
    the whole-day rule.
    """
    n_missing = trades["dt"].isna().sum()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing} trades have no timestamp; "
            "the whole-day rule needs one on every trade."
        )


CLUSTERING_RULES = {
    "match_id (primary)": cluster_by_match_id,
    "whole day": cluster_by_day,
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap(trades, cluster_labels, n_resamples=N_RESAMPLES, seed=RANDOM_SEED):
    """Resample whole clusters with replacement and recompute both ROIs.

    Resampling single trades would treat several trades from one match as
    independent evidence and give an interval that is too narrow.

    The trades are first reduced to one row per cluster, holding the total
    stake, total profit, sum of trade returns and number of trades. Each
    resample draws n_clusters clusters with replacement and recomputes flat
    ROI (sum of returns divided by number of trades) and stake-weighted ROI
    (total profit divided by total stake) from those totals.

    Returns a dict with two arrays of length n_resamples, "flat" and
    "stake_weighted".
    """
    # One row per cluster.
    per_cluster = trades.groupby(cluster_labels).agg(
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        return_sum=("trade_return", "sum"),
        n_trades=("trade_return", "count"),
    )
    stake = per_cluster["stake"].to_numpy()
    profit = per_cluster["profit"].to_numpy()
    return_sum = per_cluster["return_sum"].to_numpy()
    n_trades = per_cluster["n_trades"].to_numpy()
    n_clusters = len(per_cluster)

    # Each row of `picks` is one resample of cluster positions, drawn with
    # replacement.
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    # Totals over the drawn clusters, one resample per row.
    flat = return_sum[picks].sum(axis=1) / n_trades[picks].sum(axis=1)
    stake_weighted = profit[picks].sum(axis=1) / stake[picks].sum(axis=1)

    return {"flat": flat, "stake_weighted": stake_weighted}


def summarise_bootstrap(resampled_rois):
    """Summarise resampled ROIs as a 95% percentile interval and the share at or below zero."""
    return {
        "ci_low": np.percentile(resampled_rois, 2.5),
        "ci_high": np.percentile(resampled_rois, 97.5),
        "p_at_or_below_zero": np.mean(resampled_rois <= 0),
    }


def robustness_table(core, n_resamples=N_RESAMPLES, seed=RANDOM_SEED):
    """Run the cluster bootstrap under each clustering rule, one row per rule.

    Agreement between the rules shows the result does not depend on how
    trades are grouped.
    """
    rows = []
    for rule_name, rule in CLUSTERING_RULES.items():
        labels = rule(core)
        resampled = cluster_bootstrap(core, labels, n_resamples, seed)
        flat = summarise_bootstrap(resampled["flat"])
        stake_weighted = summarise_bootstrap(resampled["stake_weighted"])
        rows.append({
            "rule": rule_name,
            "clusters": labels.nunique(),
            "flat_ci_low": flat["ci_low"],
            "flat_ci_high": flat["ci_high"],
            "flat_p_at_or_below_zero": flat["p_at_or_below_zero"],
            "stake_ci_low": stake_weighted["ci_low"],
            "stake_ci_high": stake_weighted["ci_high"],
            "stake_p_at_or_below_zero": stake_weighted["p_at_or_below_zero"],
        })
    return pd.DataFrame(rows).set_index("rule")


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------

def competition_table(trades):
    """Summarise every trade in the file by competition.

    Unlike the core sample, this includes first-try trades and voids, so the
    counts, stakes and profits add up to the whole file. Both ROIs exclude
    voids.
    """
    rows = []
    for name, group in trades.groupby(competition_names(trades)):
        rows.append(summarise_competition(name, group))
    rows.sort(key=lambda row: row["trades"], reverse=True)
    rows.append(summarise_competition("All", trades))
    return pd.DataFrame(rows).set_index("competition")


def summarise_competition(name, group):
    """Totals and ROIs for one competition."""
    settled = group[group["result"] != "void"]
    return {
        "competition": name,
        "trades": len(group),
        "voids": len(group) - len(settled),
        "staked": group["stake"].sum(),
        "profit": group["profit"].sum(),
        "profit_units": group["profit_units"].sum(),
        "flat_roi": flat_roi(settled),
        "stake_weighted_roi": stake_weighted_roi(settled),
    }


def odds_band_table(core):
    """Compare the win rate with the implied probability in each odds band.

    If the prices were fair, the win rate in each band would match the
    implied probability. A higher win rate means the price was too long.
    """
    band = pd.cut(core["payout_odds"], bins=ODDS_BAND_EDGES, labels=ODDS_BAND_LABELS)
    rows = []
    for band_label, group in core.groupby(band, observed=True):
        rows.append(summarise_odds_band(band_label, group))
    rows.append(summarise_odds_band("All", core))
    return pd.DataFrame(rows).set_index("odds band")


def summarise_odds_band(label, group):
    """Win rate, implied probability and ROI for one odds band."""
    win_rate = group["won"].mean()
    implied = group["implied_prob"].mean()
    return {
        "odds band": label,
        "trades": len(group),
        "implied_prob": implied,
        "win_rate": win_rate,
        "win_minus_implied": win_rate - implied,
        "flat_roi": flat_roi(group),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def percent(x):
    return f"{x:+.1%}"


def interval(low, high):
    return f"{percent(low)} to {percent(high)}"


def dollars(x):
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def print_core_summary(core, primary):
    print("CORE SAMPLE: NRL Premiership, try_n >= 2, voids excluded")
    print(f"  trades               {len(core)}")
    print(f"  clusters (match_id)  {int(primary['clusters'])}")
    print(f"  flat ROI             {percent(flat_roi(core))}")
    print(f"  stake-weighted ROI   {percent(stake_weighted_roi(core))}")
    print(f"  win rate             {core['won'].mean():.1%}")
    print(f"  implied probability  {core['implied_prob'].mean():.1%}")
    print()
    print(f"MATCH-CLUSTERED BOOTSTRAP: {N_RESAMPLES:,} resamples, "
          f"clusters = match_id, seed {RANDOM_SEED}")
    print("  ROI              95% CI             P(ROI <= 0)")
    print(f"  flat             {interval(primary['flat_ci_low'], primary['flat_ci_high']):<19}"
          f"{primary['flat_p_at_or_below_zero']:.3f}")
    print(f"  stake-weighted   {interval(primary['stake_ci_low'], primary['stake_ci_high']):<19}"
          f"{primary['stake_p_at_or_below_zero']:.3f}")


def print_robustness_table(table):
    print("ROBUSTNESS: the same bootstrap under each clustering rule")
    display = pd.DataFrame({
        "clusters": table["clusters"],
        "flat 95% CI": [interval(lo, hi) for lo, hi in
                        zip(table["flat_ci_low"], table["flat_ci_high"])],
        "flat P(<=0)": table["flat_p_at_or_below_zero"].map("{:.3f}".format),
        "stake-wtd 95% CI": [interval(lo, hi) for lo, hi in
                             zip(table["stake_ci_low"], table["stake_ci_high"])],
        "stake-wtd P(<=0)": table["stake_p_at_or_below_zero"].map("{:.3f}".format),
    })
    print(display.to_string())


def print_competition_table(table):
    print("BY COMPETITION: every trade, including try_n = 1 and voids")
    print("(ROIs exclude voids; 1 unit = $400)")
    display = pd.DataFrame({
        "trades": table["trades"],
        "voids": table["voids"],
        "staked": table["staked"].map(dollars),
        "profit": table["profit"].map(dollars),
        "profit (units)": table["profit_units"].map("{:+.1f}".format),
        "flat ROI": table["flat_roi"].map(percent),
        "stake-wtd ROI": table["stake_weighted_roi"].map(percent),
    })
    print(display.to_string())


def print_odds_band_table(table):
    print("WIN RATE VS IMPLIED PROBABILITY, by odds band (core sample)")
    display = pd.DataFrame({
        "trades": table["trades"],
        "implied": table["implied_prob"].map("{:.1%}".format),
        "win rate": table["win_rate"].map("{:.1%}".format),
        "win - implied": table["win_minus_implied"].map(lambda x: f"{x * 100:+.1f} pp"),
        "flat ROI": table["flat_roi"].map(percent),
    })
    print(display.to_string())


def main():
    parser = argparse.ArgumentParser(description="Evaluate the live next-try trades.")
    parser.add_argument("--trades", default=DEFAULT_TRADES_PATH,
                        help="path to trades.csv (default: data/trades.csv)")
    args = parser.parse_args()

    trades = load_trades(args.trades)
    core = core_sample(trades)

    # The primary interval is the match_id row of the robustness table.
    robustness = robustness_table(core)
    primary = robustness.loc["match_id (primary)"]

    print_core_summary(core, primary)
    print()
    print_robustness_table(robustness)
    print()
    print_competition_table(competition_table(trades))
    print()
    print_odds_band_table(odds_band_table(core))


if __name__ == "__main__":
    main()
