"""Allow ``python -m benchmarks`` from the repository root."""

from __future__ import annotations

from benchmarks.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
