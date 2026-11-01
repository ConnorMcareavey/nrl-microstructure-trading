"""Draw the README figures into figures/.

Blue is always the 2013-2019 history and orange is always the bets, in every
figure. Run from the repository root, after the scrape and parse, with

    python -m nrl_microstructure.figures
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from nrl_microstructure import clustering, evaluate, history, odds, reopening
from nrl_microstructure.parse import PROCESSED_DIR


FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"

# Colours for the background, text, gridlines and the two series. Checked for
# colour-blind readers and for contrast against the background.
THEME = {"surface": "#fcfcfb", "text": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
         "grid": "#e1e0d9", "axis": "#c3c2b7", "history": "#2a78d6", "bets": "#eb6834",
         "inplay": "#6250d6"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
})

LINE_WIDTH = 1.8
MARKER_SIZE = 7
WIDTH, HEIGHT = 7.5, 4.2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """Everything the figures need, computed exactly as in history.py and evaluate.py."""
    matches = pd.read_csv(PROCESSED_DIR / "matches.csv")
    tries = pd.read_csv(PROCESSED_DIR / "tries.csv")
    home_win_prob = odds.home_win_probability_from_odds(matches, odds.load_odds())
    nrl = matches[(matches["competition"] == "nrl") & matches["tries_ok"]
                  & matches["match_key"].map(home_win_prob).notna()]

    moments = history.historical_moments(tries, nrl, home_win_prob)
    table = history.lookup_table(moments)
    bets = history.core_bets(matches, tries)
    bets = bets[bets["match_key"].map(home_win_prob).notna()].sort_values("dt")
    bet_moments = history.moment_before_each_bet(bets, tries, home_win_prob)
    chance = history.chance_for_each_bet(table, bets, bet_moments)

    per_match = clustering.per_match_table(tries, matches)
    seasons = per_match.groupby("season").agg(tries_per_game=("n_tries", "mean"),
                                              observed=("observed", "sum"),
                                              transitions=("transitions", "sum"))
    seasons["same_team_next"] = seasons["observed"] / seasons["transitions"]

    data = {
        "bets": bets,
        "chance": chance,
        "predicted_resamples": history.predicted_return_resamples(moments, bets, bet_moments),
        "luck_resamples": evaluate.cluster_bootstrap(bets, bets["match_id"])["flat"],
        "seasons": seasons,
    }

    # In-play prices, if reopening.py has been run. Men's World Cup only.
    if (reopening.CACHE_DIR / "reopening_curves.csv").exists():
        curves = pd.read_csv(reopening.CACHE_DIR / "reopening_curves.csv")
        rows = pd.read_csv(reopening.CACHE_DIR / "reopenings.csv")
        games = reopening.match_to_nrl_com(reopening.recorded_games(), matches)
        men = curves[curves["league"] == "World Cup"]
        data["inplay_curves"] = men
        data["inplay_rows"] = rows[rows["league"] == "World Cup"]
        data["inplay_drift"] = reopening.drift_curve(men)
        data["traded_reopenings"] = reopening.traded_reopenings(games, matches, tries, horizon=500)
    return data


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

def new_figure(theme, ncols=1, size=(WIDTH, HEIGHT), nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=theme["surface"], squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(theme["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["axis"])
        ax.tick_params(colors=theme["muted"], length=0, labelsize=9)
        ax.grid(axis="y", color=theme["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
    return fig, axes.flatten()


def add_titles(fig, theme, title, subtitle):
    # Positioned in inches from the top, so taller figures keep the same spacing.
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / height, title, ha="left", va="top", fontsize=13, fontweight="semibold",
             color=theme["text"])
    fig.text(0.02, 1 - 0.44 / height, subtitle, ha="left", va="top", fontsize=9.5, color=theme["secondary"],
             linespacing=1.4)


def percent_axis(axis, decimals=0):
    axis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=decimals))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def price_vs_history(data, theme):
    """Figure 1. Implied probability against the 2013-2019 lookup table probability, for the same trades."""
    bets, chance = data["bets"], data["chance"]
    groups = pd.qcut(bets["implied_prob"], 7, labels=False)
    by_group = pd.DataFrame({"implied": bets["implied_prob"], "history": chance}).groupby(groups).mean()

    fig, (ax,) = new_figure(theme, size=(6.2, 5.6))
    low, high = 0.25, 0.75
    ax.plot([low, high], [low, high], color=theme["muted"], linewidth=1)
    ax.text(0.62, 0.595, "fair odds", color=theme["muted"], fontsize=9, ha="left", va="top",
            rotation=45, rotation_mode="anchor", transform_rotates_text=True)
    ax.plot(by_group["implied"], by_group["history"], color=theme["history"], linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE, markeredgecolor=theme["surface"], markeredgewidth=1.5)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    percent_axis(ax.xaxis)
    percent_axis(ax.yaxis)
    ax.set_xlabel("Implied probability (1 / odds)", color=theme["secondary"])
    ax.set_ylabel("2013-2019 lookup table probability", color=theme["secondary"])
    ax.text(0.27, 0.685, "Above the line, the\nodds were too long", color=theme["secondary"], fontsize=9,
            ha="left", va="top")
    add_titles(fig, theme, "Lookup table probability against implied probability",
               "The 703 trades in seven groups of about 100 by implied probability.\n"
               "Each point is the group's average probability from the 2013-2019\n"
               "lookup table.")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.76, bottom=0.11)
    return fig


def predicted_vs_actual(data, theme):
    """Figure 2. Predicted and actual flat ROI, with 95% CIs."""
    bets = data["bets"]
    predicted = history.predicted_return(data["chance"], bets)
    rows = [
        ("Predicted\n(2013-2019 table)", predicted, np.percentile(data["predicted_resamples"], [2.5, 97.5]), theme["history"]),
        ("Actual\n(703 trades)", bets["trade_return"].mean(), np.percentile(data["luck_resamples"], [2.5, 97.5]), theme["bets"]),
    ]
    fig, (ax,) = new_figure(theme)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.6)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(0, color=theme["axis"], linewidth=1)
    ax.text(0.002, 1.62, "ROI = 0", color=theme["muted"], fontsize=9, ha="left", va="center")
    for y, (label, value, (low, high), color) in zip([1, 0], rows):
        ax.plot([low, high], [y, y], color=color, linewidth=LINE_WIDTH, solid_capstyle="round")
        ax.plot(value, y, "o", color=color, markersize=MARKER_SIZE + 2, markeredgecolor=theme["surface"], markeredgewidth=1.5)
        ax.text(value, y + 0.22, f"{value:+.1%}", color=theme["text"], fontsize=10, fontweight="semibold", ha="center")
        ax.text(high + 0.004, y, f"95% CI {low:+.1%} to {high:+.1%}", color=theme["secondary"], fontsize=9, va="center")
    ax.set_yticks([1, 0], [r[0] for r in rows], color=theme["text"], fontsize=10)
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlim(-0.02, 0.30)
    percent_axis(ax.xaxis)
    add_titles(fig, theme, "Predicted and actual ROI",
               "Flat ROI on the 703 trades. The prediction uses each trade's odds and the 2013-2019 lookup\n"
               "table, not its result. Its CI resamples historical matches; the actual CI resamples trades by game.")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.76, bottom=0.10)
    return fig


def profit_over_season(data, theme):
    """Figure 3. Cumulative actual and expected profit through the 2026 season."""
    bets = data["bets"]
    expected_profit = bets["stake"] * (data["chance"] * bets["payout_odds"] - 1)
    actual = bets["profit"].cumsum()
    predicted = expected_profit.cumsum()

    fig, (ax,) = new_figure(theme)
    ax.axhline(0, color=theme["axis"], linewidth=1)
    ax.plot(bets["dt"], predicted, color=theme["history"], linewidth=LINE_WIDTH, label="Expected from the 2013-2019 table")
    ax.plot(bets["dt"], actual, color=theme["bets"], linewidth=LINE_WIDTH, label="Actual")
    for series, color in [(predicted, theme["history"]), (actual, theme["bets"])]:
        ax.plot(bets["dt"].iloc[-1], series.iloc[-1], "o", color=color, markersize=MARKER_SIZE,
                markeredgecolor=theme["surface"], markeredgewidth=1.5)
        ax.text(bets["dt"].iloc[-1] + pd.Timedelta(days=4), series.iloc[-1], f"${series.iloc[-1]:,.0f}",
                color=theme["text"], fontsize=9.5, va="center")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}K" if v else "$0"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_xlim(bets["dt"].iloc[0] - pd.Timedelta(days=5), bets["dt"].iloc[-1] + pd.Timedelta(days=30))
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=theme["secondary"])
    add_titles(fig, theme, "Cumulative profit, 2026 season",
               "The 703 trades in date order. Expected profit per trade is stake × (lookup table probability × odds − 1).\n"
               "Stakes increased through the season, so both lines steepen.")
    fig.subplots_adjust(left=0.09, right=0.90, top=0.76, bottom=0.10)
    return fig


def statistical_significance(data, theme):
    """Figure 4. Bootstrap distribution of flat ROI, resampling trades by game."""
    resamples = data["luck_resamples"]
    actual = data["bets"]["trade_return"].mean()

    fig, (ax,) = new_figure(theme)
    ax.hist(resamples, bins=np.arange(-0.06, 0.30, 0.005), color=theme["bets"], edgecolor=theme["surface"], linewidth=1)
    ax.axvline(0, color=theme["text"], linewidth=1)
    ax.text(0.003, ax.get_ylim()[1] * 0.92, "ROI = 0", color=theme["secondary"], fontsize=9, ha="left")
    ax.text(actual, ax.get_ylim()[1] * 1.02, f"observed ROI {actual:+.1%}", color=theme["text"], fontsize=9.5,
            fontweight="semibold", ha="center")
    share_at_loss = np.mean(resamples <= 0)
    ax.text(-0.055, ax.get_ylim()[1] * 0.55, f"P(ROI ≤ 0) = {share_at_loss:.3f}",
            color=theme["secondary"], fontsize=9, ha="left")
    ax.set_yticks([])
    percent_axis(ax.xaxis)
    ax.set_xlim(-0.06, 0.30)
    add_titles(fig, theme, "Statistical significance",
               "Bootstrap distribution of flat ROI on the 703 trades, from 20,000 resamples of whole games.\n"
               "Trades in the same game are resampled together because they are not independent.")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.74, bottom=0.10)
    return fig


def rule_change(data, theme):
    """Figure 5. Tries per game and the share of next tries scored by the same team, by season."""
    seasons = data["seasons"]
    fig, (left, right) = new_figure(theme, ncols=2)
    for ax, column, title, fmt in [(left, "tries_per_game", "Tries per game", None),
                                   (right, "same_team_next", "Same team scored the next try", "percent")]:
        ax.axvspan(2019.5, 2020.5, color=theme["grid"], linewidth=0)
        ax.plot(seasons.index, seasons[column], color=theme["history"], linewidth=LINE_WIDTH,
                marker="o", markersize=MARKER_SIZE - 2, markeredgecolor=theme["surface"], markeredgewidth=1.2)
        ax.set_title(title, loc="left", color=theme["text"], fontsize=10.5, pad=8)
        ax.set_xticks([2013, 2016, 2019, 2022, 2026])
        if fmt == "percent":
            ax.set_ylim(0.45, 0.65)
            ax.set_yticks([0.45, 0.50, 0.55, 0.60, 0.65])
            percent_axis(ax.yaxis)
        else:
            ax.set_ylim(6, 9)
        ax.text(2020, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "six-again\nbegins",
                color=theme["muted"], fontsize=8, ha="center", va="bottom")
    add_titles(fig, theme, "Tries per game and same-team next try, by season",
               "NRL Premiership, 2013-2026. Six-again was introduced partway through 2020 (shaded). Tries per game\n"
               "rose afterwards, while the share of next tries scored by the same team stayed near 55%.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.10, wspace=0.25)
    return fig


SHORT_NAMES = {"Australia": "AUS", "New Zealand": "NZ", "Papua New Guinea": "PNG", "Lebanon": "LEB",
               "England": "ENG", "Tonga": "TGA", "Samoa": "SAM", "France": "FRA", "Fiji": "FIJ", "Cook Islands": "COK"}


def reopening_drift(data, theme):
    """Figure 6. Change in the scorer's probability after the Nth Try market reopens, men's World Cup."""
    curves, drift, rows = data["inplay_curves"], data["inplay_drift"], data["inplay_rows"]
    seconds = [c for c in curves.columns if str(c).isdigit()]
    x = [int(c) for c in seconds]

    fig, (ax,) = new_figure(theme)
    for _, row in curves.iterrows():
        ax.plot(x, row[seconds].astype(float) * 100, color=theme["inplay"], linewidth=0.6, alpha=0.15)
    ax.fill_between(drift["seconds"], drift["low"] * 100, drift["high"] * 100, color=theme["inplay"],
                    alpha=0.18, linewidth=0)
    ax.plot(drift["seconds"], drift["mean"] * 100, color=theme["inplay"], linewidth=LINE_WIDTH + 0.4)
    ax.axhline(0, color=theme["axis"], linewidth=1)

    median_first = rows["secs_until_first_change"].median()
    ax.axvline(median_first, color=theme["muted"], linewidth=1)
    ax.text(median_first + 4, 9.2, f"median first change\nin the opening price, {median_first:.0f} s",
            color=theme["secondary"], fontsize=9, va="top")
    at_120 = drift.loc[drift["seconds"] == 120].iloc[0]
    ax.plot(120, at_120["mean"] * 100, "o", color=theme["inplay"], markersize=MARKER_SIZE,
            markeredgecolor=theme["surface"], markeredgewidth=1.5)
    ax.text(124, at_120["mean"] * 100 - 1.2, f"mean {at_120['mean'] * 100:+.1f} pts at 120 s\n"
            f"(95% CI {at_120['low'] * 100:+.1f} to {at_120['high'] * 100:+.1f})",
            color=theme["text"], fontsize=9, va="top")

    ax.set_xlim(0, 240)
    ax.set_ylim(-8, 10)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f} pts" if v else "0"))
    ax.set_xlabel("Seconds after the market reopened", color=theme["secondary"])
    add_titles(fig, theme, "Scorer's probability after the next-try market reopens",
               f"Men's World Cup 2026, {rows['game'].nunique()} games, {len(rows)} reopenings. Change in the margin-free "
               "probability of the team that just\nscored. Faint lines are individual reopenings; the line and band are "
               "the mean and 95% CI (games resampled).")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.12)
    return fig


def traded_reopenings(data, theme):
    """Figure 7. The backed team's odds after reopening, for each traded World Cup reopening."""
    traded = data["traded_reopenings"]
    ncols = 4
    nrows = -(-len(traded) // ncols)
    fig, axes = new_figure(theme, ncols=ncols, nrows=nrows, size=(9.6, 2.35 * nrows + 1.3))
    for ax, t in zip(axes, traded):
        xs = t["seconds"] + [t["end_secs"]]
        ys = t["odds"] + [t["odds"][-1]]
        suspended = t["suspended"] + [t["suspended"][-1]]
        # Shade the stretches where the market was suspended, and draw their
        # (stale) quotes faintly so they don't set the scale.
        for start, stop, is_suspended in zip(xs[:-1], xs[1:], suspended):
            if is_suspended:
                ax.axvspan(start, stop, color=theme["grid"], linewidth=0)
        ax.step(xs, ys, where="post", color=theme["muted"], linewidth=0.8)
        open_ys = [np.nan if s else y for y, s in zip(ys, suspended)]
        ax.step(xs, open_ys, where="post", color=theme["inplay"], linewidth=LINE_WIDTH)
        if t["secs_to_trade"] <= t["end_secs"]:
            ax.plot(t["secs_to_trade"], t["odds_taken"], "o", color=theme["bets"], markersize=MARKER_SIZE,
                    markeredgecolor=theme["surface"], markeredgewidth=1.5)
        open_values = [y for y in open_ys if not np.isnan(y)] + [t["odds_taken"]]
        low, high = min(open_values), max(open_values)
        pad = max(0.04, 0.15 * (high - low))
        ax.set_ylim(low - pad, high + pad)
        home, away = t["game"].split(" v ")
        ax.set_title(f"{SHORT_NAMES.get(home, home)} v {SHORT_NAMES.get(away, away)}, try {t['try_number']}, "
                     f"backed {SHORT_NAMES.get(t['team'], t['team'])}", loc="left", color=theme["text"], fontsize=9)
        ax.set_xlim(0, max(240, t["end_secs"]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    for ax in axes[len(traded):]:
        ax.set_visible(False)
    add_titles(fig, theme, "The traded reopenings",
               "Men's World Cup 2026. The backed team's odds from the moment the next-try market reopened, for at least "
               "four minutes and past the trade.\nThe orange dot is the trade, grey shading marks suspensions, and the "
               "x-axis is seconds after reopening.")
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.06, right=0.98, top=1 - 1.45 / height, bottom=0.05, hspace=0.55, wspace=0.28)
    return fig


FIGURES = {
    "1-price-vs-history": price_vs_history,
    "2-predicted-vs-actual": predicted_vs_actual,
    "3-profit-over-season": profit_over_season,
    "4-statistical-significance": statistical_significance,
    "5-rule-change": rule_change,
    "6-reopening-drift": reopening_drift,
    "7-traded-reopenings": traded_reopenings,
}


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    data = load_data()
    for name, draw in FIGURES.items():
        if name.startswith(("6-", "7-")) and "inplay_curves" not in data:
            continue      # in-play figures need reopening.py to have been run
        fig = draw(data, THEME)
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, facecolor=THEME["surface"])
        plt.close(fig)
        print(f"figures/{name}.png")


if __name__ == "__main__":
    main()
