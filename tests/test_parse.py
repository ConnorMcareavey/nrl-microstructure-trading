"""Tests for nrl_microstructure.parse, on small hand-built match records.

The records have the same shape as nrl.com's but are made up, so no scraped
data is stored in the repository.
"""

from nrl_microstructure.parse import parse_match


def make_match(timeline, home_tries=2, away_tries=1, home_score=12, away_score=6):
    return {
        "matchId": "1",
        "roundNumber": 1,
        "roundTitle": "Round 1",
        "startTime": "2024-03-03T02:30:00Z",
        "venue": "Somewhere",
        "homeTeam": {"teamId": 1, "nickName": "Home", "score": home_score,
                     "scoring": {"tries": {"made": home_tries}}},
        # nrl.com gives None, not 0, for a team with no tries.
        "awayTeam": {"teamId": 2, "nickName": "Away", "score": away_score,
                     "scoring": {"tries": {"made": away_tries or None}}},
        "timeline": timeline,
    }


# Home 12, Away 6.
#   10:00 Home try (4-0), conversion (6-0)
#   30:00 Away try (6-4), conversion (6-6)
#   50:00 Home try (10-6), conversion (12-6)
# The 50:00 try is listed before the 30:00 one on purpose, to check the sort.
GOOD_TIMELINE = [
    {"type": "GameTime", "title": "KICK OFF", "gameSeconds": 0, "teamId": 0},
    {"type": "Try", "gameSeconds": 600, "teamId": 1, "homeScore": 4},
    {"type": "Goal", "gameSeconds": 640, "teamId": 1, "homeScore": 6},
    {"type": "Try", "gameSeconds": 3000, "teamId": 1, "homeScore": 10, "awayScore": 6},
    {"type": "Try", "gameSeconds": 1800, "teamId": 2, "homeScore": 6, "awayScore": 4},
    {"type": "Goal", "gameSeconds": 1840, "teamId": 2, "homeScore": 6, "awayScore": 6},
    {"type": "Goal", "gameSeconds": 3040, "teamId": 1, "homeScore": 12, "awayScore": 6},
]


def test_tries_come_out_in_match_order_with_running_scores():
    _, tries = parse_match("nrl", 2024, "2024_round-1_x", make_match(GOOD_TIMELINE))

    assert [t["try_number"] for t in tries] == [1, 2, 3]
    assert [t["team"] for t in tries] == ["Home", "Away", "Home"]
    assert [t["game_seconds"] for t in tries] == [600, 1800, 3000]
    # A missing score key means that side had 0.
    assert (tries[0]["home_score_after"], tries[0]["away_score_after"]) == (4, 0)


def test_a_consistent_match_passes_the_check():
    match_row, _ = parse_match("nrl", 2024, "2024_round-1_x", make_match(GOOD_TIMELINE))
    assert match_row["timeline_ok"]
    assert match_row["check_note"] == ""


def test_a_missing_try_fails_the_check():
    # Drop the away try from the timeline. The official count still says 1.
    timeline = [e for e in GOOD_TIMELINE if not (e["type"] == "Try" and e["teamId"] == 2)]
    match_row, _ = parse_match("nrl", 2024, "2024_round-1_x", make_match(timeline))
    assert not match_row["timeline_ok"]
    assert match_row["check_note"].startswith("try count mismatch")


def test_a_wrong_final_score_fails_the_check():
    match_row, _ = parse_match(
        "nrl", 2024, "2024_round-1_x", make_match(GOOD_TIMELINE, home_score=14)
    )
    assert not match_row["timeline_ok"]
    assert match_row["check_note"].startswith("score mismatch")


def test_a_team_with_no_tries_counts_as_zero():
    # Home 8, Away 0, from two unconverted home tries.
    match = make_match(
        [
            {"type": "Try", "gameSeconds": 600, "teamId": 1, "homeScore": 4},
            {"type": "Try", "gameSeconds": 3000, "teamId": 1, "homeScore": 8},
        ],
        home_tries=2, away_tries=0, home_score=8, away_score=0,
    )
    match_row, _ = parse_match("nrl", 2024, "2024_round-1_x", match)
    assert match_row["away_tries_official"] == 0
    assert match_row["timeline_ok"]
