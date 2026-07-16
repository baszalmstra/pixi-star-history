from datetime import date

from star_history.collector import (
    StarRow,
    anchored_history_rows,
    apply_daily_snapshot,
    read_rows,
    write_rows,
)


def test_anchored_history_uses_event_shape_and_hits_exact_anchors() -> None:
    rows = anchored_history_rows(
        "example/project",
        {date(2024, 1, 1): 1, date(2024, 1, 3): 3},
        {date(2024, 1, 3): 8},
        current_stars=10,
        through=date(2024, 1, 5),
    )

    assert [(row.day.isoformat(), row.stars, row.observation) for row in rows] == [
        ("2024-01-01", 2, "estimated"),
        ("2024-01-02", 2, "estimated"),
        ("2024-01-03", 8, "wayback"),
        ("2024-01-04", 9, "estimated"),
        ("2024-01-05", 10, "snapshot"),
    ]


def test_anchored_history_can_represent_net_unstars() -> None:
    rows = anchored_history_rows(
        "example/project",
        {},
        {date(2024, 1, 1): 10, date(2024, 1, 3): 8},
        current_stars=8,
        through=date(2024, 1, 3),
    )

    assert [row.stars for row in rows] == [10, 9, 8]
    assert rows[1].daily_change == -1


def test_daily_snapshot_replaces_today_and_uses_previous_observation() -> None:
    rows = [
        StarRow(date(2024, 1, 1), "example/project", 10, 10, "reconstructed"),
        StarRow(date(2024, 1, 3), "example/project", 12, 2, "snapshot"),
    ]

    updated = apply_daily_snapshot(rows, "example/project", 11, date(2024, 1, 3))

    assert updated[-1] == StarRow(date(2024, 1, 3), "example/project", 11, 1, "snapshot")


def test_csv_round_trip_is_sorted(tmp_path) -> None:
    path = tmp_path / "nested" / "history.csv"
    rows = [
        StarRow(date(2024, 1, 2), "b/repo", 2, 1, "snapshot"),
        StarRow(date(2024, 1, 1), "a/repo", 1, 1, "reconstructed"),
    ]

    write_rows(rows, path)

    assert read_rows(path) == list(reversed(rows))
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "date,repository,stars,daily_change,observation"
    )
