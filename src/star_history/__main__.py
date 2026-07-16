"""Command-line entry point for Pixi tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .collector import DEFAULT_CSV, collect


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="star_history")
    commands = command_parser.add_subparsers(dest="command", required=True)

    collect_parser = commands.add_parser("collect", help="Refresh the local star-history CSV")
    collect_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    collect_parser.add_argument(
        "--full",
        action="store_true",
        help="Re-fetch every current stargazer timestamp instead of taking a daily snapshot",
    )

    build_parser = commands.add_parser("build-site", help="Build the static visualization")
    build_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    build_parser.add_argument("--output", type=Path, default=Path("site/index.html"))
    return command_parser


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "collect":
        print(collect(arguments.csv, full=arguments.full))
    elif arguments.command == "build-site":
        from .site import build_site

        print(build_site(arguments.csv, arguments.output))


if __name__ == "__main__":
    main()
