"""Command-line entry for the honest rextio-numpy benchmark suite.

Usage (from repository root, with the project ``.venv`` active)::

    python -m benchmarks --output-dir /tmp/rextio-numpy-bench
    python -m benchmarks --list
    python -m benchmarks --output-dir ./benchmarks/results --samples 7

No ``pyproject.toml`` entry-point is required; the package is imported from
the repository root via ``python -m benchmarks``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.runner import RunnerConfig, run_suite, write_reports
from benchmarks.scenarios import registered_scenarios


def _repo_root() -> Path:
    # benchmarks/cli.py → benchmarks/ → repo root
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=(
            "Honest, reproducible rextio-numpy benchmark suite. "
            "Measures fallback vs native wall time in separate subprocesses; "
            "never asserts an expected speedup; reports native losses honestly."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for JSON and Markdown reports. Required unless --list. "
            "Prefer a temp path or an ignored directory (see benchmarks/.gitignore)."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of wall-time samples per leg (default: 5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Iterations per sample (default: 50).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="Untimed warmup calls per leg (default: 3).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario id to run (repeatable). Default: all registered scenarios.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered scenarios and exit.",
    )
    parser.add_argument(
        "--keep-fixture",
        action="store_true",
        help="Keep the temporary fixture project (writes under a temp dir still).",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="Optional explicit fixture project directory (created if missing).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python executable for measurement subprocesses (default: current).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for spec in registered_scenarios():
            labels = ",".join(spec.labels)
            print(f"{spec.id:28}  {spec.name}  [{labels}]")
        return 0

    if args.output_dir is None:
        parser.error("--output-dir is required (reports must not be committed)")

    if args.samples < 1 or args.iterations < 1 or args.warmups < 0:
        parser.error("--samples/--iterations must be >= 1; --warmups must be >= 0")

    config = RunnerConfig(
        output_dir=Path(args.output_dir),
        samples=int(args.samples),
        iterations=int(args.iterations),
        warmups=int(args.warmups),
        scenario_ids=list(args.scenarios) if args.scenarios else None,
        python_executable=str(args.python) if args.python else None,
        repo_root=_repo_root(),
        keep_fixture=bool(args.keep_fixture),
        fixture_dir=Path(args.fixture_dir) if args.fixture_dir else None,
    )

    try:
        report = run_suite(config)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_path, md_path = write_reports(report, config.output_dir)
    print(f"suite_status={report.suite_status}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    fixture_path = report.settings.get("fixture_dir")
    if fixture_path and report.settings.get("keep_fixture"):
        print(f"fixture_dir={fixture_path} (preserved)")
    for s in report.scenarios:
        speed = "n/a" if s.speedup is None else f"{s.speedup:.4f}x"
        print(f"  {s.id}: status={s.status} speedup={speed}")
        if s.reason:
            print(f"    reason: {s.reason}")

    # Non-successful suite → non-zero exit (honest CI signal).
    if report.suite_status == "ok":
        return 0
    if report.suite_status == "partial":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
