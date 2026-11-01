"""Pre-match win probabilities from historical head-to-head odds.

Source: the AusSportsBetting historical NRL file
(https://www.aussportsbetting.com/data/). Its terms allow personal use only,
with no redistribution, so it is kept in data/external/, which git ignores.
Convert the downloaded .xlsx once with

    python -m nrl_microstructure.odds path/to/nrl.xlsx

The prices include the bookmaker's margin, so 1/home + 1/away comes to
slightly more than 1. Dividing each by their sum removes it.

    p_home = (1/home odds) / (1/home odds + 1/away odds)

Draws are rare enough in the NRL to ignore.
"""

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ODDS_CSV = REPO_ROOT / "data" / "external" / "aussportsbetting_nrl.csv"

# AusSportsBetting team name -> nrl.com nickName. Some clubs appear under two
# spellings in different seasons.
TEAM_NAMES = {
    "Brisbane Broncos": "Broncos",
    "Canberra Raiders": "Raiders",
    "Canterbury Bulldogs": "Bulldogs",
    "Canterbury-Bankstown Bulldogs": "Bulldogs",
    "Cronulla Sharks": "Sharks",
    "Cronulla-Sutherland Sharks": "Sharks",
    "Dolphins": "Dolphins",
    "Gold Coast Titans": "Titans",
    "Manly Sea Eagles": "Sea Eagles",
    "Manly-Warringah Sea Eagles": "Sea Eagles",
    "Melbourne Storm": "Storm",
    "New Zealand Warriors": "Warriors",
    "Newcastle Knights": "Knights",
    "North QLD Cowboys": "Cowboys",
    "North Queensland Cowboys": "Cowboys",
    "Parramatta Eels": "Eels",
    "Penrith Panthers": "Panthers",
    "South Sydney Rabbitohs": "Rabbitohs",
    "St George Dragons": "Dragons",
    "St. George Illawarra Dragons": "Dragons",
    "Sydney Roosters": "Roosters",
    "Wests Tigers": "Wests Tigers",
}

# How many days apart the odds file's date and nrl.com's kickoff date may be.
# One day allows for time zone differences around midnight.
DATE_TOLERANCE_DAYS = 1


def load_odds(path=ODDS_CSV):
    """Read the converted odds CSV and add nrl.com team names and p_home."""
    odds = pd.read_csv(path, parse_dates=["Date"])
    odds["home_team"] = odds["Home Team"].map(TEAM_NAMES)
    odds["away_team"] = odds["Away Team"].map(TEAM_NAMES)
    unknown = sorted(set(odds.loc[odds["home_team"].isna(), "Home Team"])
                     | set(odds.loc[odds["away_team"].isna(), "Away Team"]))
    if unknown:
        raise ValueError(f"odds file has team names with no nrl.com mapping: {unknown}")

    home_inverse = 1 / odds["Home Odds"]
    away_inverse = 1 / odds["Away Odds"]
    odds["p_home"] = home_inverse / (home_inverse + away_inverse)
    return odds


def home_win_probability_from_odds(matches, odds):
    """Pre-match home win probability for each nrl.com match, with the margin removed.

    The two files are matched on the pair of teams and a kickoff date within
    DATE_TOLERANCE_DAYS. If the odds file lists the teams the other way round
    (at a neutral venue, for example), its probability is flipped. Returns a
    Series indexed by match_key, with NaN where no odds row was found.
    """
    # Index the odds by the unordered pair of teams.
    by_pair = {}
    for row in odds.dropna(subset=["p_home"]).itertuples():
        pair = frozenset([row.home_team, row.away_team])
        by_pair.setdefault(pair, []).append((row.Date, row.home_team, row.p_home))

    probabilities = {}
    kickoff_dates = pd.to_datetime(matches["kickoff_utc"], utc=True).dt.tz_convert("Australia/Sydney")
    for match, kickoff in zip(matches.itertuples(), kickoff_dates):
        kickoff_date = pd.Timestamp(kickoff.date())
        candidates = [
            (date, home, p) for date, home, p in by_pair.get(frozenset([match.home_team, match.away_team]), [])
            if abs((date - kickoff_date).days) <= DATE_TOLERANCE_DAYS
        ]
        if len(candidates) != 1:
            probabilities[match.match_key] = float("nan")
            continue
        _, odds_home_team, p_home = candidates[0]
        probabilities[match.match_key] = p_home if odds_home_team == match.home_team else 1 - p_home
    return pd.Series(probabilities, name="p_home_odds")


def convert_xlsx(xlsx_path, csv_path=ODDS_CSV):
    """Convert the downloaded workbook's Data sheet to CSV. Its header is on row 2."""
    data = pd.read_excel(xlsx_path, sheet_name="Data", header=1)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    print(f"{len(data)} rows -> {csv_path}")


if __name__ == "__main__":
    convert_xlsx(sys.argv[1])
