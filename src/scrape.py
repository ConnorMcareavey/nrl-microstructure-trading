"""Download finished NRL matches from nrl.com and save them locally.

Each match centre page embeds the full match record as JSON. This script
saves that record, one gzipped file per match, under
data/raw/<competition>/<season>/. It does no analysis. parse.py reads only
from these files, so the analysis can be rerun without contacting nrl.com
again.

Two things are left out of the saved files. Unfinished matches are skipped,
so a match is never saved half-played. The highlight-video details attached
to each timeline event (thumbnails, stream links and asset IDs) are removed,
because they make up most of each record and say nothing about when tries
were scored. Everything else is saved exactly as served. The full 2013-2026
download is about 33 MB for 2,920 matches.

Examples

    python src/scrape.py --seasons 2013-2026
    python src/scrape.py --competition nrlw --seasons 2026
    python src/scrape.py --seasons 2024 --delay 0.5
"""

import argparse
import gzip
import html
import json
import re
import sys
import time
from pathlib import Path

import requests

# nrl.com's internal competition IDs.
COMPETITIONS = {
    "nrl": 111,            # NRL Premiership (men's)
    "nrlw": 161,           # NRLW Premiership (women's)
    "origin": 116,         # State of Origin (men's)
    "womens-origin": 156,  # State of Origin (women's)
    "world-cup": 131,      # Rugby League World Cup
}

BASE = "https://www.nrl.com"

# The scraper identifies itself and links to this repository. Requests are
# made one at a time, with a pause after each (see --delay).
HEADERS = {
    "User-Agent": "nrl-try-momentum/0.1 (research scraper; https://github.com/ConnorMcareavey/nrl-try-momentum)"
}

# Each match centre page stores its match record as HTML-escaped JSON in the
# q-data attribute of the vue-match-centre element.
MATCH_BLOB = re.compile(r'id="vue-match-centre"[^>]*?q-data="(.*?)"', re.S)

# Only matches in this state are saved. Upcoming and live matches are skipped.
FINISHED = "FullTime"


def get(url, delay):
    """Fetch a URL, pause afterwards, and retry up to twice on a network error."""
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            time.sleep(delay)          # pause between requests
            return response.text
        except requests.RequestException as error:
            if attempt == 2:
                raise
            # Wait 2 seconds before the first retry and 6 before the second.
            print(f"    retry {attempt + 1} after {error}", file=sys.stderr)
            time.sleep(2 + 4 * attempt)


def fetch_round_numbers(competition_id, season, delay):
    """Return the round numbers in a season, finals included.

    The draw lists them in its round menu. The number of rounds differs
    between seasons and competitions, so it is read from the draw rather than
    assumed. State of Origin has no round menu and lists the whole series at
    once, so [None] is returned to mean a single request for the season.
    """
    url = f"{BASE}/draw/data?competition={competition_id}&season={season}"
    draw = json.loads(get(url, delay))
    rounds = sorted(entry["value"] for entry in draw.get("filterRounds") or [])
    return rounds or [None]


def fetch_round_fixtures(competition_id, season, round_number, delay):
    """Return the matches in one round, or in the whole season if round_number is None."""
    url = f"{BASE}/draw/data?competition={competition_id}&season={season}"
    if round_number is not None:
        url += f"&round={round_number}"
    draw = json.loads(get(url, delay))
    # Keep matches only, not byes or promotional entries.
    return [f for f in draw.get("fixtures", []) if f.get("type") == "Match"]


def extract_match(page_html):
    """Return the match record embedded in a match centre page, or None if it isn't there."""
    found = MATCH_BLOB.search(page_html)
    if not found:
        return None
    # The attribute is HTML-escaped, so it is unescaped before parsing.
    return json.loads(html.unescape(found.group(1)))["match"]


def strip_highlight_videos(match):
    """Remove the highlight-video details from each timeline event."""
    for event in match.get("timeline") or []:
        event.pop("content", None)
    return match


def slug_for(match_centre_url):
    """File name for a match, taken from its match centre URL.

    '/draw/nrl-premiership/2020/round-1/eels-v-bulldogs/' becomes
    '2020_round-1_eels-v-bulldogs', and
    '/draw/nrl-premiership/2026/grand-final/game-1/' becomes
    '2026_grand-final_game-1'.

    The slug is known from the draw before the match page is fetched, so a
    rerun can skip saved matches without requesting them. It includes the
    round, so two meetings of the same teams in a season never share a name.
    """
    parts = match_centre_url.strip("/").split("/")
    return "_".join(parts[2:])          # drop "draw" and the competition name


def cache_path(data_dir, competition, season, slug):
    # Each competition has its own folder, because an NRL and an NRLW match
    # can have the same slug.
    return Path(data_dir) / "raw" / competition / str(season) / f"{slug}.json.gz"


def scrape_season(competition, season, data_dir, delay, refresh):
    """Save every finished match in a season, skipping matches already saved."""
    competition_id = COMPETITIONS[competition]
    saved = skipped = unfinished = 0

    for round_number in fetch_round_numbers(competition_id, season, delay):
        fixtures = fetch_round_fixtures(competition_id, season, round_number, delay)

        for fixture in fixtures:
            slug = slug_for(fixture["matchCentreUrl"])
            path = cache_path(data_dir, competition, season, slug)

            # Skip matches already saved, before making a request.
            if path.exists() and not refresh:
                skipped += 1
                continue

            # Skip matches that haven't finished.
            if fixture.get("matchState") != FINISHED:
                unfinished += 1
                continue

            page = get(BASE + fixture["matchCentreUrl"], delay)
            match = extract_match(page)
            if match is None:
                print(f"  {season} R{round_number}: no match blob for {slug}", file=sys.stderr)
                continue
            if match.get("matchState") != FINISHED:
                unfinished += 1
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file and rename it once complete, so an
            # interrupted run never leaves a partial file that looks saved.
            partial = path.with_name(path.name + ".part")
            with gzip.open(partial, "wt", encoding="utf-8") as handle:
                json.dump(strip_highlight_videos(match), handle)
            partial.replace(path)
            saved += 1

        round_label = "all rounds" if round_number is None else f"R{round_number}"
        print(f"  {season} {round_label}: {len(fixtures)} fixtures")

    return saved, skipped, unfinished


def parse_seasons(text):
    """Read a single season such as '2024' or a range such as '2013-2025'."""
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(text)]


def main():
    parser = argparse.ArgumentParser(description="Save finished NRL matches from nrl.com.")
    parser.add_argument("--competition", default="nrl", choices=sorted(COMPETITIONS))
    parser.add_argument("--seasons", default="2013-2026",
                        help="a season such as 2024, or a range such as 2013-2026")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds to wait between requests")
    parser.add_argument("--refresh", action="store_true",
                        help="download matches again even if already saved")
    args = parser.parse_args()

    for season in parse_seasons(args.seasons):
        print(f"{args.competition} season {season}")
        saved, skipped, unfinished = scrape_season(
            args.competition, season, args.data_dir, args.delay, args.refresh
        )
        print(f"  -> {saved} saved, {skipped} already cached, {unfinished} not finished (skipped)")


if __name__ == "__main__":
    main()
