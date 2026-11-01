"""Turn the saved match records into tables.

Reads only from data/raw/ (written by src/scrape.py) and writes two CSVs to
data/processed/.

    matches.csv  one row per match, with teams, kickoff, final score and try counts
    tries.csv    one row per try, in match order, with who scored, when, and the score after it

Every match is checked against its official result. A match is used in the
analysis only if its timeline holds exactly the official number of tries for
each team (tries_ok). A stricter check, which also requires the running score
to finish on the final score (timeline_ok), is reported but not used to
exclude matches. Matches that fail stay in matches.csv with the reason, so
they can be counted.

Run from the repository root with

    python -m nrl_microstructure.parse
"""

import gzip
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def iter_cached_matches(raw_dir=RAW_DIR):
    """Yield (competition, season, slug, match) for every saved match file.

    Files are stored as raw/<competition>/<season>/<slug>.json.gz.
    """
    for path in sorted(Path(raw_dir).glob("*/*/*.json.gz")):
        competition = path.parent.parent.name
        season = int(path.parent.name)
        slug = path.name.removesuffix(".json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield competition, season, slug, json.load(handle)


def official_try_count(team):
    """The try count from the team's scoring summary, where None means 0."""
    tries = (team.get("scoring") or {}).get("tries") or {}
    return tries.get("made") or 0


def parse_match(competition, season, slug, match):
    """Turn one match record into (match_row, try_rows)."""
    match_key = f"{competition}_{slug}"
    home, away = match["homeTeam"], match["awayTeam"]
    team_names = {home["teamId"]: home["nickName"], away["teamId"]: away["nickName"]}
    timeline = match.get("timeline") or []

    # Tries in match order. The sort is stable, so two tries logged in the
    # same second keep the order they were served in.
    try_events = [event for event in timeline if event.get("type") == "Try"]
    try_events.sort(key=lambda event: event["gameSeconds"])

    try_rows = []
    for try_number, event in enumerate(try_events, start=1):
        try_rows.append({
            "match_key": match_key,
            "competition": competition,
            "season": season,
            "try_number": try_number,
            "game_seconds": event["gameSeconds"],
            "team_id": event["teamId"],
            "team": team_names.get(event["teamId"]),
            "is_home": event["teamId"] == home["teamId"],
            # The score straight after the try, before the conversion. A side
            # with no points yet has no score key, so a missing key means 0.
            "home_score_after": event.get("homeScore", 0),
            "away_score_after": event.get("awayScore", 0),
        })

    home_tries_timeline = sum(1 for row in try_rows if row["is_home"])
    away_tries_timeline = len(try_rows) - home_tries_timeline

    # Scores only go up, so the highest score on any event is the final score.
    # This avoids listing every scoring event type by name.
    home_score_timeline = max((e.get("homeScore", 0) for e in timeline), default=0)
    away_score_timeline = max((e.get("awayScore", 0) for e in timeline), default=0)

    match_row = {
        "match_key": match_key,
        "competition": competition,
        "season": season,
        "slug": slug,
        "match_id": match.get("matchId"),
        "round_number": match.get("roundNumber"),
        "round_title": match.get("roundTitle"),
        "kickoff_utc": match.get("startTime"),
        "venue": match.get("venue"),
        "home_team_id": home["teamId"],
        "home_team": home["nickName"],
        "away_team_id": away["teamId"],
        "away_team": away["nickName"],
        "home_score": home.get("score"),
        "away_score": away.get("score"),
        "home_tries_official": official_try_count(home),
        "away_tries_official": official_try_count(away),
        "home_tries_timeline": home_tries_timeline,
        "away_tries_timeline": away_tries_timeline,
        "home_score_timeline": home_score_timeline,
        "away_score_timeline": away_score_timeline,
        "n_events": len(timeline),
    }
    # The rule that decides whether a match is used. Every try must belong to
    # a known team, and the timeline's try counts must equal the official ones.
    match_row["tries_ok"] = (
        all(row["team"] is not None for row in try_rows)
        and home_tries_timeline == match_row["home_tries_official"]
        and away_tries_timeline == match_row["away_tries_official"]
    )
    match_row["timeline_ok"], match_row["check_note"] = check_match(match_row, try_rows)
    return match_row, try_rows


def check_match(match_row, try_rows):
    """The stricter check. Returns (ok, reason), with an empty reason when the match passes.

    The timeline must have events, every try must belong to a known team, the
    try counts must equal the official ones, and the running score must finish
    on the final score.
    """
    if match_row["n_events"] == 0:
        return False, "no timeline"
    if any(row["team"] is None for row in try_rows):
        return False, "try credited to an unknown team"
    if (match_row["home_tries_timeline"] != match_row["home_tries_official"]
            or match_row["away_tries_timeline"] != match_row["away_tries_official"]):
        return False, (
            f"try count mismatch: timeline {match_row['home_tries_timeline']}-"
            f"{match_row['away_tries_timeline']}, official "
            f"{match_row['home_tries_official']}-{match_row['away_tries_official']}"
        )
    if (match_row["home_score_timeline"] != match_row["home_score"]
            or match_row["away_score_timeline"] != match_row["away_score"]):
        return False, (
            f"score mismatch: timeline {match_row['home_score_timeline']}-"
            f"{match_row['away_score_timeline']}, final "
            f"{match_row['home_score']}-{match_row['away_score']}"
        )
    return True, ""


def parse_all(raw_dir=RAW_DIR):
    """Parse every saved match into two DataFrames, matches and tries."""
    match_rows, try_rows = [], []
    for competition, season, slug, match in iter_cached_matches(raw_dir):
        match_row, tries = parse_match(competition, season, slug, match)
        match_rows.append(match_row)
        try_rows.extend(tries)
    return pd.DataFrame(match_rows), pd.DataFrame(try_rows)


def main():
    matches, tries = parse_all()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_DIR / "matches.csv", index=False)
    tries.to_csv(PROCESSED_DIR / "tries.csv", index=False)

    print(f"{len(matches)} matches, {len(tries)} tries")
    print(f"{(~matches['tries_ok']).sum()} matches failed the try-count check and are excluded from the analysis")
    failed = matches[~matches["timeline_ok"]]
    print(f"{len(failed)} matches failed the stricter check (try counts or final score)")
    for _, row in failed.iterrows():
        print(f"  {row['match_key']}: {row['check_note']}")


if __name__ == "__main__":
    main()
