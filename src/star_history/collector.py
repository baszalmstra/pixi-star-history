"""Collect estimated historical star activity and exact daily GitHub snapshots."""

from __future__ import annotations

import csv
import os
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPOSITORIES = ("conda/conda", "prefix-dev/pixi", "mamba-org/mamba")
REPOSITORY_ALIASES = {
    "conda/conda": ("ContinuumIO/conda", "pydata/conda", "conda/conda"),
    "prefix-dev/pixi": ("prefix-dev/pixi",),
    "mamba-org/mamba": ("QuantStack/mamba", "TheSnakePit/mamba", "mamba-org/mamba"),
}
CSV_FIELDS = ("date", "repository", "stars", "daily_change", "observation")
DEFAULT_CSV = Path("data/star_history.csv")
CLICKHOUSE_URL = "https://play.clickhouse.com/?user=play"


@dataclass(frozen=True, order=True)
class StarRow:
    """One repository's cumulative star count on one UTC date."""

    day: date
    repository: str
    stars: int
    daily_change: int
    observation: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "date": self.day.isoformat(),
            "repository": self.repository,
            "stars": self.stars,
            "daily_change": self.daily_change,
            "observation": self.observation,
        }


def retrying_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class GitHubClient:
    """Authenticated GitHub GraphQL client for aggregate star counts."""

    def __init__(self, token: str) -> None:
        self.session = retrying_session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "pixi-star-history",
            }
        )

    def current_counts(self, repositories: Iterable[str]) -> dict[str, int]:
        """Fetch current public stargazer totals in one GraphQL call."""
        repositories = tuple(repositories)
        declarations = ", ".join(
            f"$owner{i}: String!, $name{i}: String!" for i in range(len(repositories))
        )
        selections = "\n".join(
            f"repo{i}: repository(owner: $owner{i}, name: $name{i}) "
            "{ stargazerCount nameWithOwner }"
            for i in range(len(repositories))
        )
        query = f"query StarCounts({declarations}) {{\n{selections}\n}}"
        variables: dict[str, str] = {}
        for index, repository in enumerate(repositories):
            owner, name = repository.split("/", 1)
            variables[f"owner{index}"] = owner
            variables[f"name{index}"] = name

        response = self.session.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": variables},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if errors := payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL returned errors: {errors}")

        counts: dict[str, int] = {}
        for index, repository in enumerate(repositories):
            result = payload["data"][f"repo{index}"]
            if result is None:
                raise RuntimeError(f"GitHub repository not found: {repository}")
            counts[repository] = int(result["stargazerCount"])
        return counts


class GitHubArchiveClient:
    """Query the public ClickHouse mirror of GH Archive."""

    def __init__(self) -> None:
        self.session = retrying_session()
        self.session.headers.update({"User-Agent": "pixi-star-history"})

    def daily_watch_events(self) -> dict[str, Counter[date]]:
        """Return daily WatchEvent counts, accounting for known repository renames."""
        alias_to_repository = {
            alias.casefold(): repository
            for repository, aliases in REPOSITORY_ALIASES.items()
            for alias in aliases
        }
        quoted_aliases = ", ".join(f"'{alias.casefold()}'" for alias in alias_to_repository)
        query = f"""
            SELECT lower(repo_name) AS repository_alias,
                   toDate(created_at) AS day,
                   count() AS stars_added
            FROM github_events
            WHERE event_type = 'WatchEvent'
              AND lower(repo_name) IN ({quoted_aliases})
            GROUP BY repository_alias, day
            ORDER BY repository_alias, day
            FORMAT JSON
        """
        response = self.session.post(CLICKHOUSE_URL, data=query, timeout=180)
        response.raise_for_status()

        events = {repository: Counter() for repository in REPOSITORIES}
        for item in response.json()["data"]:
            repository = alias_to_repository[item["repository_alias"].casefold()]
            events[repository][date.fromisoformat(item["day"])] += int(item["stars_added"])
        return events


def resolve_token() -> str:
    """Resolve authentication without ever writing the token to disk."""
    for variable in ("GITHUB_TOKEN", "GH_TOKEN", "STAR_HISTORY_PAT"):
        if token := os.environ.get(variable):
            return token

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "Set GITHUB_TOKEN, or authenticate the GitHub CLI with `gh auth login`."
        ) from error
    return result.stdout.strip()


def anchored_history_rows(
    repository: str,
    daily_events: Mapping[date, int],
    archived_anchors: Mapping[date, int],
    current_stars: int,
    through: date,
) -> list[StarRow]:
    """Interpolate daily totals between aggregate anchors using star-event activity."""
    points = {
        day: stars for day, stars in archived_anchors.items() if day <= through and stars >= 0
    }
    points[through] = current_stars
    ordered_points = sorted(points.items())
    if not ordered_points:
        return []

    first_day = min(daily_events, default=ordered_points[0][0])
    first_day = min(first_day, ordered_points[0][0])
    rows: list[StarRow] = []
    previous_stars = 0

    segment_start_day = first_day - timedelta(days=1)
    segment_start_stars = 0
    for segment_end_day, segment_end_stars in ordered_points:
        if segment_end_day < first_day:
            continue
        days = (segment_end_day - segment_start_day).days
        weighted_total = sum(
            daily_events.get(segment_start_day + timedelta(days=offset), 0)
            for offset in range(1, days + 1)
        )
        weighted_progress = 0
        for offset in range(1, days + 1):
            day = segment_start_day + timedelta(days=offset)
            weighted_progress += daily_events.get(day, 0)
            progress = weighted_progress / weighted_total if weighted_total else offset / days
            stars = segment_start_stars + round(
                (segment_end_stars - segment_start_stars) * progress
            )
            if day == through:
                observation = "snapshot"
            elif day in archived_anchors:
                observation = "wayback"
            else:
                observation = "estimated"
            rows.append(StarRow(day, repository, stars, stars - previous_stars, observation))
            previous_stars = stars
        segment_start_day = segment_end_day
        segment_start_stars = segment_end_stars

    return rows


def apply_daily_snapshot(
    rows: Iterable[StarRow], repository: str, stars: int, day: date
) -> list[StarRow]:
    """Insert or replace today's exact total while retaining historical rows."""
    result = [row for row in rows if not (row.repository == repository and row.day == day)]
    previous = max(
        (row for row in result if row.repository == repository and row.day < day),
        key=lambda row: row.day,
        default=None,
    )
    change = stars - previous.stars if previous else stars
    result.append(StarRow(day, repository, stars, change, "snapshot"))
    return sorted(result, key=lambda row: (row.day, row.repository))


def read_rows(path: Path = DEFAULT_CSV) -> list[StarRow]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            StarRow(
                day=date.fromisoformat(item["date"]),
                repository=item["repository"],
                stars=int(item["stars"]),
                daily_change=int(item["daily_change"]),
                observation=item["observation"],
            )
            for item in csv.DictReader(handle)
        ]


def write_rows(rows: Iterable[StarRow], path: Path = DEFAULT_CSV) -> None:
    """Atomically write a deterministically ordered CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        ordered = sorted(rows, key=lambda row: (row.day, row.repository))
        writer.writerows(row.as_dict() for row in ordered)
    temporary.replace(path)


def collect(path: Path = DEFAULT_CSV, *, full: bool = False, today: date | None = None) -> str:
    """Initialize estimated history or append a cheap exact daily snapshot."""
    today = today or datetime.now(UTC).date()
    counts = GitHubClient(resolve_token()).current_counts(REPOSITORIES)

    if full or not path.exists():
        from .wayback import WaybackClient, write_anchors

        archived_events = GitHubArchiveClient().daily_watch_events()
        anchors = WaybackClient().anchors()
        write_anchors(anchors)
        anchors_by_repository = {
            repository: {
                anchor.day: anchor.stars for anchor in anchors if anchor.repository == repository
            }
            for repository in REPOSITORIES
        }
        rows = [
            row
            for repository in REPOSITORIES
            for row in anchored_history_rows(
                repository,
                archived_events[repository],
                anchors_by_repository[repository],
                counts[repository],
                today,
            )
        ]
        mode = "Wayback-anchored GH Archive backfill"
    else:
        rows = read_rows(path)
        for repository, stars in counts.items():
            rows = apply_daily_snapshot(rows, repository, stars, today)
        mode = "GraphQL daily snapshot"

    write_rows(rows, path)
    return f"Wrote {len(rows):,} rows to {path} using {mode}."
