"""Unified command line interface for live website enrichment."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tavily_crawler",
        description="Probe website liveness or run the live Tavily crawl.",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("liveness", add_help=False, help="Probe and update website_alive")
    commands.add_parser("crawl", add_help=False, help="Run the resumable Tavily crawl")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command, *command_args = args
    if command == "liveness":
        from .liveness import main as liveness_main

        liveness_main(command_args)
        return
    if command == "crawl":
        from .crawl_cli import main as crawl_main

        crawl_main(command_args)
        return

    parser.error(f"unknown command: {command}")
