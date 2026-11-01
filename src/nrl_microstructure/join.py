"""Link each trade to its match and to the tries either side of it.

For every row of trades.csv
1. Find the match in the same competition where the backed team kicked off
   no more than four hours before the trade. A team plays at most once in
   that window, so there is never more than one candidate.
2. Find the tries either side of the trade. The market for try k opens after
   try k - 1 is scored, so a trade on try_n = k sits between those two tries.
3. Check the link against the trade's own result. A trade that won must have
   had try k scored by the backed team, and a trade that lost must not.

Check 3 is the strongest evidence that a link is right, because it uses
information step 1 never looked at. Every settled trade that links to a match
passes it. Timestamps in trades.csv are Brisbane time (UTC+10, no daylight
saving).
"""

import pandas as pd


TRADES_TIMEZONE = "Australia/Brisbane"

# A trade must come no more than this many minutes after its match's kickoff.
MAX_MINUTES_AFTER_KICKOFF = 240

# Allowance for the timing check. Half-time and stoppages put real time ahead
# of game time, so a trade is only flagged if it comes more than this many
# minutes after the next try's game time.
WALL_CLOCK_SLACK_MINUTES = 45

# trades.csv team name -> nrl.com nickName, where they differ.
NAME_FIXES = {"Manly": "Sea Eagles", "Tigers": "Wests Tigers"}

# Origin sides, as written in trades.csv -> (competition, nrl.com nickName).
# (nrl.com calls the NSW women's side "Blues Women" but the QLD one "Maroons".)
ORIGIN_TEAMS = {
    "NSW (Origin)": ("origin", "Blues"),
    "QLD (Origin)": ("origin", "Maroons"),
    "NSW (W)": ("womens-origin", "Blues Women"),
    "QLD (W)": ("womens-origin", "Maroons"),
}


def nrl_com_team(comp, team):
    """Map a trades.csv (comp, team) to (nrl.com competition, nickName).

    Returns (None, None) where nrl.com has no data, such as Super League.
    """
    if team in ORIGIN_TEAMS:
        return ORIGIN_TEAMS[team]
    if comp == "nrl":
        return "nrl", NAME_FIXES.get(team, team)
    if comp == "nrlw":
        return "nrlw", NAME_FIXES.get(team.removesuffix(" (W)"), team.removesuffix(" (W)"))
    if comp == "international":
        return "world-cup", team
    return None, None


def link_trade(trade, matches, tries_by_match):
    """Link one trade. Returns a dict of what was found and the results of the checks."""
    competition, nickname = nrl_com_team(trade["comp"], trade["team"])
    link = {"match_key": None, "minutes_after_kickoff": None, "prev_try_team": None,
            "next_try_team": None, "backed_just_scored": None, "result_consistent": None,
            "timing_consistent": None, "link_note": ""}

    if competition is None:
        link["link_note"] = "no nrl.com data for this competition"
        return link
    if pd.isna(trade["dt"]):
        link["link_note"] = "no timestamp"
        return link

    placed_at = trade["dt"].tz_localize(TRADES_TIMEZONE).tz_convert("UTC")
    candidates = matches[(matches["competition"] == competition)
                         & ((matches["home_team"] == nickname) | (matches["away_team"] == nickname))]
    minutes = (placed_at - candidates["kickoff_utc"]).dt.total_seconds() / 60
    in_window = minutes[(minutes >= 0) & (minutes <= MAX_MINUTES_AFTER_KICKOFF)]
    if in_window.empty:
        link["link_note"] = f"no {competition} match for {nickname} in the 4 hours before the trade"
        return link

    # The most recent kickoff before the trade.
    match = candidates.loc[in_window.idxmin()]
    link["match_key"] = match["match_key"]
    link["minutes_after_kickoff"] = float(in_window.min())

    match_tries = tries_by_match.get(match["match_key"], pd.DataFrame(columns=["try_number"]))
    k = trade["try_n"]
    prev_try = match_tries[match_tries["try_number"] == k - 1]
    next_try = match_tries[match_tries["try_number"] == k]

    if len(prev_try):
        link["prev_try_team"] = prev_try.iloc[0]["team"]
        link["backed_just_scored"] = link["prev_try_team"] == nickname
    if len(next_try):
        link["next_try_team"] = next_try.iloc[0]["team"]

    # Check 3. The trade's result against who actually scored try k.
    backed_scored_next = link["next_try_team"] == nickname
    if trade["result"] == "won":
        link["result_consistent"] = backed_scored_next
    elif trade["result"] == "lost":
        link["result_consistent"] = not backed_scored_next

    # Timing check. Real time never runs behind game time, so the trade must
    # come after try k-1's game time, and not long after try k's.
    kickoff = match["kickoff_utc"]
    after_prev = (not len(prev_try)) or placed_at >= kickoff + pd.Timedelta(seconds=int(prev_try.iloc[0]["game_seconds"]))
    before_next = (not len(next_try)) or placed_at <= kickoff + pd.Timedelta(
        seconds=int(next_try.iloc[0]["game_seconds"])) + pd.Timedelta(minutes=WALL_CLOCK_SLACK_MINUTES)
    link["timing_consistent"] = bool(after_prev and before_next)
    return link


def link_trades(trades, matches, tries):
    """Link every trade. Returns the trades with the link columns added."""
    matches = matches.copy()
    matches["kickoff_utc"] = pd.to_datetime(matches["kickoff_utc"], utc=True)
    tries_by_match = {key: group for key, group in tries.groupby("match_key")}
    links = [link_trade(trade, matches, tries_by_match) for _, trade in trades.iterrows()]
    return pd.concat([trades.reset_index(drop=True), pd.DataFrame(links)], axis=1)
