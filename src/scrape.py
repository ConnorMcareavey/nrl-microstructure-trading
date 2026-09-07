"""
Stage 1: fetch NRL match data from nrl.com and cache it on disk.

Design notes (the reasoning matters as much as the code here):

*Why cache at all.* Every downstream result must be reproducible without a
network connection, and reviewable by someone who does not trust our parser.
So this script does one thing only -- pull bytes down and store them. It makes
no analytic decisions. Parsing lives in parse.py, which reads only from the
cache. That separation means a parsing bug can be fixed and the whole analysis
rerun in seconds, without re-scraping or hitting nrl.com again.

*Why not store the raw HTML.* The match centre page is ~300 KB, mostly markup
and video metadata. Thirteen seasons of it is roughly 780 MB, which is not
something you can sensibly commit. The page embeds all the match data we care
about as a JSON blob, so we store that instead.

*What we drop, and why that is still honest.* Each timeline event carries a
`content` object describing the highlight video for that moment -- thumbnails,
stream URLs, asset ids. It is the bulk of the payload and has no bearing on
when a try was scored. Dropping it takes the cache from ~260 MB to ~24 MB.
Nothing else is removed: the full event timeline, both teams, the venue and
the match stats are stored exactly as served. The rule is deliberately a
single line so it is easy to state and easy to audit.

Usage:
    python src/scrape.py --seasons 2013-2025
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

# nrl.com's internal id for the Telstra Premiership (the top-grade men's comp).
NRL_PREMIERSHIP = 111

BASE = "https://www.nrl.com"

# nrl.com serves different markup to unrecognised clients, so identify as a
# normal browser. We are reading public pages at a slow, fixed rate.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# The match centre is a Vue app. The server renders its initial state into a
# `q-data` attribute as HTML-escaped JSON. This is the whole match record.
MATCH_BLOB = re.compile(r'id="vue-match-centre"[^>]*?q-data="(.*?)"', re.S)

# Rounds are numbered 1..27ish for the regular season, then finals continue the
# numbering. We do not know the count in advance and it varies by season (2020
# was shortened), so we walk upward and stop once several rounds in a row come
# back empty.
MAX_ROUND = 32
EMPTY_ROUNDS_BEFORE_STOPPING = 3


def get(url, delay):
    """Fetch a URL politely, with a short retry on transient failures."""
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            time.sleep(delay)          # rate limit ourselves, always
            return response.text
        except requests.RequestException as error:
            if attempt == 2:
                raise
            # Back off progressively: 2s, then 6s.
            print(f"    retry {attempt + 1} after {error}", file=sys.stderr)
            time.sleep(2 + 4 * attempt)


def fetch_round_fixtures(season, round_number, delay):
    """Return the fixture list for one round, as served by the draw endpoint."""
    url = f"{BASE}/draw/data?competition={NRL_PREMIERSHIP}&season={season}&round={round_number}"
    draw = json.loads(get(url, delay))
    # The draw also contains byes and promotional entries; only keep real games.
    return [f for f in draw.get("fixtures", []) if f.get("type") == "Match"]


def extract_match(page_html):
    """Pull the embedded match record out of a match centre page."""
    found = MATCH_BLOB.search(page_html)
    if not found:
        return None
    # The attribute is HTML-escaped (&quot; for every quote), so unescape first.
    return json.loads(html.unescape(found.group(1)))["match"]


def strip_highlight_videos(match):
    """Remove the highlight-video payload from each timeline event. See module docstring."""
    for event in match.get("timeline") or []:
        event.pop("content", None)
    return match


def slug_for(match_centre_url):
    """
    Turn '/draw/nrl-premiership/2020/round-1/eels-v-bulldogs/' into
    '2020_round-1_eels-v-bulldogs'.

    We key the cache on this slug rather than on nrl.com's numeric matchId,
    because the slug is known from the draw *before* we fetch the match page.
    That is what lets a re-run skip cached matches without spending a request
    on them. It is also readable: you can tell what a cached file holds from
    its name, which matters for a repo meant to be audited. The round number
    is part of the slug, so two meetings of the same teams never collide.
    """
    parts = match_centre_url.strip("/").split("/")
    return "_".join(parts[2:])          # drop the 'draw/nrl-premiership' prefix


def cache_path(data_dir, season, slug):
    return Path(data_dir) / "raw" / str(season) / f"{slug}.json.gz"


def scrape_season(season, data_dir, delay, refresh):
    """Fetch every match in a season, skipping anything already cached."""
    saved = skipped = 0
    consecutive_empty = 0

    for round_number in range(1, MAX_ROUND + 1):
        fixtures = fetch_round_fixtures(season, round_number, delay)

        if not fixtures:
            consecutive_empty += 1
            if consecutive_empty >= EMPTY_ROUNDS_BEFORE_STOPPING:
                break
            continue
        consecutive_empty = 0

        for fixture in fixtures:
            slug = slug_for(fixture["matchCentreUrl"])
            path = cache_path(data_dir, season, slug)

            # Check the cache before spending a request.
            if path.exists() and not refresh:
                skipped += 1
                continue

            page = get(BASE + fixture["matchCentreUrl"], delay)
            match = extract_match(page)
            if match is None:
                print(f"  {season} R{round_number}: no match blob for {slug}", file=sys.stderr)
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            # Store gzipped: these files compress ~10x and are meant to be committed.
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(strip_highlight_videos(match), handle)
            saved += 1

        print(f"  {season} R{round_number}: {len(fixtures)} fixtures")

    return saved, skipped


def parse_seasons(text):
    """Accept either '2024' or a '2013-2025' range."""
    if "-" in text:
        first, last = text.split("-")
        return list(range(int(first), int(last) + 1))
    return [int(text)]


def main():
    parser = argparse.ArgumentParser(description="Cache NRL match data locally.")
    parser.add_argument("--seasons", default="2013-2025", help="e.g. 2024 or 2013-2025")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between requests; keep this polite")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download matches that are already cached")
    args = parser.parse_args()

    for season in parse_seasons(args.seasons):
        print(f"season {season}")
        saved, skipped = scrape_season(season, args.data_dir, args.delay, args.refresh)
        print(f"  -> {saved} saved, {skipped} already cached")


if __name__ == "__main__":
    main()
