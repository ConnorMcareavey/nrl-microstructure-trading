"""How the next-try price moves after the market reopens, across every reopening recorded.

After each try the bookmaker suspends its "Nth Try" market and reopens it for
the next try. The hypothesis is that the reopening price for the team that
just scored is too long, and shortens over the following minute. This module
measures that move at every reopening in the recorded World Cup games,
whether or not it was traded.

Data. In-play price history comes from the MoviOdds historical odds API
(the /history endpoint), which has rugby league from 13 October 2026. It is
cached in data/external/inplay/ and not committed. The order of tries comes
from nrl.com, via scrape.py and parse.py.

Measure. The margin-free probability of the team that just scored,
p = (1/o_s) / (1/o_s + 1/o_o), from the latest price of both teams. For each
reopening, the change in p from the opening price to 30, 60 and 120 seconds
later. A window is dropped if the market suspends, or the next market opens,
inside it. As a control, the same change over a later window, 180 to 240
seconds after reopening. Men's games are the main result; women's games are
reported separately. Uncertainty is from resampling whole games.

Run from the repository root, after the scrape and parse, with

    python -m nrl_microstructure.reopening
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nrl_microstructure import join
from nrl_microstructure.evaluate import load_trades
from nrl_microstructure.parse import PROCESSED_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "external" / "inplay"
API = "https://moviodds.com"
HEADERS = {"User-Agent": "nrl-microstructure-trading/0.1 (research; https://github.com/ConnorMcareavey/nrl-microstructure-trading)"}

NTH_TRY_GROUP = "190242"
LEAGUES = {"World Cup": "world-cup", "World Cup Women": "womens-world-cup"}
FIRST_DAY, LAST_DAY = "2026-10-13", "2026-11-01"

HORIZONS = [30, 60, 120]          # seconds after reopening
CONTROL = (180, 240)              # a later 60-second window in the same market
CURVE_SECONDS = np.arange(0, 301, 5)
N_RESAMPLES = 5000
RANDOM_SEED = 2026


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get(path, **params):
    """Fetch from the archive, retrying up to three times on a server error."""
    for attempt in range(4):
        response = requests.get(API + path, params=params, headers=HEADERS, timeout=120)
        time.sleep(0.6)           # stay well under the 120 requests per minute limit
        if response.status_code < 500 or attempt == 3:
            response.raise_for_status()
            return response.json()
        time.sleep(5 * (attempt + 1))


def recorded_games(first_day=FIRST_DAY, last_day=LAST_DAY):
    """Finished rugby league games in the archive, one row per game, cached as CSV."""
    path = CACHE_DIR / f"games_{first_day}_{last_day}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"fixture_id": str})
    rows = []
    for day in pd.date_range(first_day, last_day):
        for game in get("/results", sport="rugby_league", date=f"{day:%Y-%m-%d}").get("results", []):
            rows.append({"fixture_id": str(game["fixture_id"]), "league": game["league"],
                         "home": game["home"], "away": game["away"], "date": f"{day:%Y-%m-%d}"})
    games = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    games.to_csv(path, index=False)
    return games


def nth_try_prices(fixture_id):
    """Every recorded price change in a game's Nth Try markets, oldest first, cached as JSON."""
    path = CACHE_DIR / f"{fixture_id}_nth_try.json"
    if not path.exists():
        data = get("/history", fi=fixture_id, group=NTH_TRY_GROUP, limit=50000)
        if data.get("truncated"):
            raise RuntimeError(f"history for {fixture_id} was truncated")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data["points"]))
    points = pd.DataFrame(json.loads(path.read_text()))
    if points.empty:
        return points
    points["ts"] = pd.to_datetime(points["ts"], format="ISO8601")
    points["try_number"] = points["group_name"].str.extract(r"(\d+)").astype(int)
    points["team"] = points["label"].str.replace(r"\s*\(W\)$", "", regex=True)
    return points.sort_values("ts", kind="stable").reset_index(drop=True)


def match_to_nrl_com(games, matches):
    """The nrl.com match_key for each archived game, matched on competition, teams and date."""
    keys = []
    kickoff_dates = pd.to_datetime(matches["kickoff_utc"], utc=True).dt.tz_convert("UTC").dt.normalize()
    for game in games.itertuples():
        competition = LEAGUES.get(game.league)
        teams = {re.sub(r"\s*\(W\)$", "", game.home), re.sub(r"\s*\(W\)$", "", game.away)}
        day = pd.Timestamp(game.date, tz="UTC")
        found = matches[(matches["competition"] == competition)
                        & matches.apply(lambda m: {m["home_team"], m["away_team"]} == teams, axis=1)
                        & ((kickoff_dates - day).abs() <= pd.Timedelta(days=1))]
        keys.append(found["match_key"].iloc[0] if len(found) == 1 else None)
    return games.assign(match_key=keys)


# ---------------------------------------------------------------------------
# Reopenings
# ---------------------------------------------------------------------------

def scorer_probability_path(market, scorer, other):
    """The margin-free probability of the scorer after each recorded point, and whether it was suspended."""
    latest = {}
    rows = []
    for point in market.itertuples():
        latest[point.team] = (point.odds_dec, point.suspended)
        if scorer in latest and other in latest:
            o_s, s_suspended = latest[scorer]
            o_o, o_suspended = latest[other]
            p = (1 / o_s) / (1 / o_s + 1 / o_o)
            rows.append({"ts": point.ts, "p": p, "odds": o_s, "suspended": s_suspended or o_suspended})
    return pd.DataFrame(rows)


def value_at(path, when):
    """The latest state at or before a time."""
    before = path[path["ts"] <= when]
    return None if before.empty else before.iloc[-1]


def window_is_clean(path, start, end, market_end):
    """No suspension inside (start, end], and the market still open at end."""
    inside = path[(path["ts"] > start) & (path["ts"] <= end)]
    return end < market_end and not inside["suspended"].any()


def reopenings_for_game(points, tries):
    """One row per reopening of an Nth Try market (N >= 2) where the previous try's scorer is known."""
    rows, curves = [], []
    scorer_of = tries.set_index("try_number")["team"]
    market_start = points.groupby("try_number")["ts"].min()
    for n, market in points.groupby("try_number"):
        if n < 2 or (n - 1) not in scorer_of.index:
            continue
        scorer = scorer_of[n - 1]
        teams = set(market["team"])
        if scorer not in teams or len(teams) != 2:
            continue
        other = (teams - {scorer}).pop()
        path = scorer_probability_path(market, scorer, other)
        open_rows = path[~path["suspended"]]
        if open_rows.empty:
            continue
        opened = open_rows.iloc[0]
        t0, p0 = opened["ts"], opened["p"]
        # The market closes at the next try. The next market opening is a
        # firm upper bound; without one, the last recorded point is used.
        market_end = market_start.get(n + 1, path["ts"].max())

        row = {"try_number": n, "scorer": scorer, "opening_odds": opened["odds"], "opening_p": p0}
        for h in HORIZONS:
            end = t0 + pd.Timedelta(seconds=h)
            row[f"change_{h}s"] = (value_at(path, end)["p"] - p0) if window_is_clean(path, t0, end, market_end) else np.nan
        start, end = (t0 + pd.Timedelta(seconds=s) for s in CONTROL)
        if window_is_clean(path, t0, end, market_end):
            row["control_change"] = value_at(path, end)["p"] - value_at(path, start)["p"]
        else:
            row["control_change"] = np.nan
        changed = path[(path["ts"] > t0) & (path["odds"] != opened["odds"])]
        row["secs_until_first_change"] = (changed["ts"].iloc[0] - t0).total_seconds() if len(changed) else np.nan
        rows.append(row)

        curve = {}
        for s in CURVE_SECONDS:
            end = t0 + pd.Timedelta(seconds=int(s))
            curve[int(s)] = (value_at(path, end)["p"] - p0) if window_is_clean(path, t0, end, market_end) or s == 0 else np.nan
        curves.append(curve)
    return pd.DataFrame(rows), pd.DataFrame(curves)


def all_reopenings(games, tries):
    rows, curves = [], []
    for game in games.dropna(subset=["match_key"]).itertuples():
        points = nth_try_prices(game.fixture_id)
        if points.empty:
            continue
        game_rows, game_curves = reopenings_for_game(points, tries[tries["match_key"] == game.match_key])
        if game_rows.empty:
            continue
        label = {"fixture_id": game.fixture_id, "league": game.league, "game": f"{game.home} v {game.away}"}
        rows.append(game_rows.assign(**label))
        curves.append(game_curves.assign(**label))
    return pd.concat(rows, ignore_index=True), pd.concat(curves, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def game_bootstrap_mean(values, games, n_resamples=N_RESAMPLES, seed=RANDOM_SEED):
    """Mean of values with a 95% CI from resampling whole games."""
    data = pd.DataFrame({"value": values, "game": games}).dropna()
    per_game = data.groupby("game")["value"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_game), size=(n_resamples, len(per_game)))
    means = per_game["sum"].to_numpy()[picks].sum(axis=1) / per_game["count"].to_numpy()[picks].sum(axis=1)
    return data["value"].mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), len(data), np.mean(means <= 0)


def drift_curve(curves, n_resamples=N_RESAMPLES, seed=RANDOM_SEED):
    """Mean change in the scorer's probability at each second after reopening, with a 95% CI from resampling games."""
    seconds = [c for c in curves.columns if str(c).isdigit()]
    games = curves["game"].to_numpy()
    unique_games = pd.unique(games)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique_games), size=(n_resamples, len(unique_games)))
    rows = []
    for s in seconds:
        per_game = (pd.DataFrame({"value": curves[s], "game": games}).dropna()
                    .groupby("game")["value"].agg(["sum", "count"]).reindex(unique_games, fill_value=0))
        sums, counts = per_game["sum"].to_numpy(), per_game["count"].to_numpy()
        resampled = sums[picks].sum(axis=1) / np.maximum(counts[picks].sum(axis=1), 1)
        rows.append({"seconds": int(s), "mean": curves[s].mean(), "n": int(curves[s].notna().sum()),
                     "low": np.percentile(resampled, 2.5), "high": np.percentile(resampled, 97.5)})
    return pd.DataFrame(rows)


def traded_reopenings(games, matches, tries, horizon=240):
    """The traded World Cup reopenings, with the backed team's odds from reopening until the next try or the horizon.

    Suspensions inside that window (for a video review, say) are kept and
    flagged, so a trade placed after the market reopened again still shows.
    """
    linked = join.link_trades(load_trades(), matches, tries)
    traded = linked[(linked["comp"] == "international") & linked["match_key"].notna()]
    by_key = games.dropna(subset=["match_key"]).set_index("match_key")
    out = []
    for trade in traded.itertuples():
        if trade.match_key not in by_key.index:
            continue
        game = by_key.loc[trade.match_key]
        points = nth_try_prices(game["fixture_id"])
        market = points[(points["try_number"] == trade.try_n) & (points["team"] == trade.team)]
        if market.empty or market["suspended"].all():
            continue
        t0 = market.loc[~market["suspended"], "ts"].iloc[0]
        placed = pd.Timestamp(trade.dt).tz_localize(join.TRADES_TIMEZONE).tz_convert("UTC")
        # Show at least four minutes, and past the trade. The archive records
        # changes only, so the last price carries forward to the end.
        shown = min(horizon, max(240, (placed - t0).total_seconds() + 30))
        next_market = points.loc[points["try_number"] == trade.try_n + 1, "ts"].min()
        last_seen = max(market["ts"].max(), placed + pd.Timedelta(seconds=30))
        stops = [t0 + pd.Timedelta(seconds=shown), next_market if pd.notna(next_market) else last_seen]
        end = min(stops)
        path = market[(market["ts"] >= t0) & (market["ts"] <= end)]
        out.append({"game": f"{game['home']} v {game['away']}", "try_number": trade.try_n, "team": trade.team,
                    "odds_taken": trade.odds, "secs_to_trade": (placed - t0).total_seconds(),
                    "end_secs": (end - t0).total_seconds(),
                    "seconds": (path["ts"] - t0).dt.total_seconds().tolist(), "odds": path["odds_dec"].tolist(),
                    "suspended": path["suspended"].tolist()})
    return out


def open_price_at(market, when):
    """The latest open (not suspended) price at or before a time."""
    open_points = market[(market["ts"] <= when) & ~market["suspended"]]
    return open_points["odds_dec"].iloc[-1] if len(open_points) else np.nan


def opening_clv(rows, horizons=(60, 120)):
    """Closing line value of the scorer's opening odds against its own price h seconds later.

    CLV = opening odds / later odds - 1, the standard measure. Both prices
    include the bookmaker's margin. A window is dropped if the market
    suspends, or the next market opens, inside it.
    """
    out = []
    for fixture_id, game_rows in rows.groupby("fixture_id"):
        points = nth_try_prices(str(fixture_id))
        starts = points.groupby("try_number")["ts"].min()
        for row in game_rows.itertuples():
            market = points[(points["try_number"] == row.try_number) & (points["team"] == row.scorer)]
            t0 = market.loc[~market["suspended"], "ts"].iloc[0]
            market_end = starts.get(row.try_number + 1, market["ts"].max())
            for h in horizons:
                when = t0 + pd.Timedelta(seconds=h)
                inside = market[(market["ts"] > t0) & (market["ts"] <= when)]
                if when >= market_end or inside["suspended"].any():
                    continue
                out.append({"game": row.game, "horizon": h, "opening_odds": row.opening_odds,
                            "clv": row.opening_odds / open_price_at(market, when) - 1})
    return pd.DataFrame(out)


def traded_clv(traded, horizons=(60, 120)):
    """Closing line value of each traded reopening: odds taken against the open price h seconds after the trade."""
    rows = []
    for t in traded:
        path = pd.DataFrame({"ts": t["seconds"], "odds_dec": t["odds"], "suspended": t["suspended"]})
        row = {"reopening": f"{t['game']}, try {t['try_number']}", "backed": t["team"],
               "secs_after_open": t["secs_to_trade"], "odds_taken": t["odds_taken"]}
        for h in horizons:
            when = t["secs_to_trade"] + h
            row[f"clv_{h}s"] = (t["odds_taken"] / open_price_at(path, when) - 1) if when <= t["end_secs"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_clv(clv, label):
    print(f"{label}: CLV of the opening odds against the bookmaker's own later price")
    for h, d in clv.groupby("horizon"):
        mean, low, high, n, _ = game_bootstrap_mean(d["clv"], d["game"])
        print(f"  +{h}s  n={n:3d}  mean {mean:+.1%} (95% CI {low:+.1%} to {high:+.1%})  median {d['clv'].median():+.1%}  "
              f"better {(d['clv'] > 0).mean():.0%}  same {(d['clv'] == 0).mean():.0%}  worse {(d['clv'] < 0).mean():.0%}")
    print()


def summarise(rows, label):
    print(f"{label}: {rows['game'].nunique()} games, {len(rows)} reopenings")
    print(f"  {'window':22s}{'n':>5s}{'mean change in p':>18s}   {'95% CI':22s}{'shorter':>8s}{'same':>7s}{'longer':>8s}{'P(mean <= 0)':>14s}")
    for name, column in [(f"open to +{h}s", f"change_{h}s") for h in HORIZONS] + [("control +180s to +240s", "control_change")]:
        mean, low, high, n, p_le = game_bootstrap_mean(rows[column], rows["game"])
        v = rows[column].dropna()
        print(f"  {name:22s}{n:5d}{mean * 100:+17.2f}pt   {low * 100:+.2f} to {high * 100:+.2f} pt{'':5s}"
              f"{(v > 0).mean():8.0%}{(v == 0).mean():7.0%}{(v < 0).mean():8.0%}{p_le:14.3f}")
    first = rows["secs_until_first_change"].dropna()
    print(f"  opening price first changed after a median of {first.median():.0f} seconds "
          f"(middle half {first.quantile(0.25):.0f} to {first.quantile(0.75):.0f})")
    # Expected return of the opening odds if the margin-free probability two
    # minutes later is the true one. Unlike CLV, this takes the margin out.
    expected = rows["opening_odds"] * (rows["opening_p"] + rows["change_120s"]) - 1
    mean, low, high, n, _ = game_bootstrap_mean(expected, rows["game"])
    print(f"  expected return of the opening odds at the +120s margin-free probability: {mean:+.1%} "
          f"(95% CI {low:+.1%} to {high:+.1%}, n={n})\n")


def main():
    matches = pd.read_csv(PROCESSED_DIR / "matches.csv")
    tries = pd.read_csv(PROCESSED_DIR / "tries.csv")
    games = match_to_nrl_com(recorded_games(), matches)
    unmatched = games[games["match_key"].isna()]
    print(f"{len(games)} archived games, {len(unmatched)} without an nrl.com match\n")
    rows, curves = all_reopenings(games, tries)
    rows.to_csv(CACHE_DIR / "reopenings.csv", index=False)
    curves.to_csv(CACHE_DIR / "reopening_curves.csv", index=False)
    men, women = rows[rows["league"] == "World Cup"], rows[rows["league"] == "World Cup Women"]
    summarise(men, "MEN'S WORLD CUP (main result)")
    summarise_clv(opening_clv(men), "MEN'S WORLD CUP")
    summarise(women, "WOMEN'S WORLD CUP (separate check)")
    summarise_clv(opening_clv(women), "WOMEN'S WORLD CUP")

    traded = traded_clv(traded_reopenings(games, matches, tries, horizon=900))
    print("TRADED REOPENINGS: odds taken against the open price after the trade")
    print(traded.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    for h in (60, 120):
        c = traded[f"clv_{h}s"].dropna()
        print(f"  +{h}s: {len(c)} with a clean window, mean CLV {c.mean():+.1%}, median {c.median():+.1%}, "
              f"better {int((c > 0).sum())}, same {int((c == 0).sum())}, worse {int((c < 0).sum())}")


if __name__ == "__main__":
    main()
