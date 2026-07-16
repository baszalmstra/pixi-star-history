"""Extract historical aggregate star-count anchors from archived GitHub pages."""

from __future__ import annotations

import csv
import re
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from .collector import REPOSITORIES, REPOSITORY_ALIASES, retrying_session

CDX_URL = "https://web.archive.org/cdx/search/cdx"
REPLAY_URL = "https://web.archive.org/web/{timestamp}id_/{original}"
ALIAS_YEARS = {
    "continuumio/conda": (2012, 2013),
    "pydata/conda": (2013, 2014),
    "conda/conda": (2014, None),
    "prefix-dev/pixi": (2023, None),
    "quantstack/mamba": (2019, 2020),
    "thesnakepit/mamba": (2020, 2020),
    "mamba-org/mamba": (2020, None),
}
DEFAULT_ANCHORS_CSV = Path("data/wayback_anchors.csv")
STAR_PATTERNS = (
    re.compile(r'aria-label=["\']([\d,]+) users? starred', re.IGNORECASE),
    re.compile(r'"stargazerCount"\s*:\s*(\d+)', re.IGNORECASE),
    re.compile(r'"stargazers_count"\s*:\s*(\d+)', re.IGNORECASE),
)


@dataclass(frozen=True, order=True)
class StarAnchor:
    """An aggregate count parsed from one archived repository page."""

    day: date
    repository: str
    stars: int
    timestamp: str
    url: str


class WaybackClient:
    """Query monthly Wayback snapshots and parse GitHub's displayed count."""

    def __init__(self, *, request_pause: float = 0.5) -> None:
        self.session = retrying_session()
        self.session.headers.update({"User-Agent": "pixi-star-history/0.1 (research project)"})
        self.request_pause = request_pause

    def snapshot_index(self, alias: str) -> list[tuple[str, str]]:
        """Return at most one successful HTML snapshot per month."""
        start_year, end_year = ALIAS_YEARS[alias.casefold()]
        parameters = [
            ("url", f"github.com/{alias}"),
            ("output", "json"),
            ("filter", "statuscode:200"),
            ("filter", "mimetype:text/html"),
            ("fl", "timestamp,original"),
            ("collapse", "timestamp:6"),
            ("from", str(start_year)),
            ("to", str(end_year or datetime.now(UTC).year)),
        ]
        response = self.session.get(CDX_URL, params=parameters, timeout=180)
        response.raise_for_status()
        payload = response.json()
        if len(payload) < 2:
            return []
        columns = payload[0]
        return [
            (item[columns.index("timestamp")], item[columns.index("original")])
            for item in payload[1:]
        ]

    def parse_snapshot(self, repository: str, timestamp: str, original: str) -> StarAnchor | None:
        url = REPLAY_URL.format(timestamp=timestamp, original=original)
        response = self.session.get(url, timeout=120)
        response.raise_for_status()
        if (stars := parse_star_count(response.text)) is None:
            return None
        return StarAnchor(
            day=datetime.strptime(timestamp[:8], "%Y%m%d").date(),
            repository=repository,
            stars=stars,
            timestamp=timestamp,
            url=url,
        )

    def anchors(self, repositories: Iterable[str] = REPOSITORIES) -> list[StarAnchor]:
        """Fetch all parseable monthly anchors for the selected repositories."""
        anchors: list[StarAnchor] = []
        for repository in repositories:
            for alias in REPOSITORY_ALIASES[repository]:
                try:
                    snapshots = self.snapshot_index(alias)
                except requests.RequestException as error:
                    warnings.warn(f"Skipping Wayback index for {alias}: {error}", stacklevel=2)
                    continue
                for timestamp, original in snapshots:
                    try:
                        anchor = self.parse_snapshot(repository, timestamp, original)
                    except requests.RequestException as error:
                        warnings.warn(
                            f"Skipping Wayback snapshot {timestamp} for {alias}: {error}",
                            stacklevel=2,
                        )
                        anchor = None
                    if anchor:
                        anchors.append(anchor)
                    time.sleep(self.request_pause)

        # Prefer the latest capture when multiple aliases or captures cover one day.
        by_day = {(anchor.repository, anchor.day): anchor for anchor in anchors}
        return sorted(by_day.values())


def parse_star_count(html: str) -> int | None:
    """Extract an exact aggregate count from one archived GitHub page."""
    for pattern in STAR_PATTERNS:
        if match := pattern.search(html):
            return int(match.group(1).replace(",", ""))
    return None


def write_anchors(anchors: Iterable[StarAnchor], path: Path = DEFAULT_ANCHORS_CSV) -> None:
    """Write the exact Wayback observations used to estimate daily history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("date", "repository", "stars", "timestamp", "url"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "date": anchor.day.isoformat(),
                "repository": anchor.repository,
                "stars": anchor.stars,
                "timestamp": anchor.timestamp,
                "url": anchor.url,
            }
            for anchor in sorted(anchors)
        )
    temporary.replace(path)
