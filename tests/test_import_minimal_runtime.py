"""Regression: rextio_numpy imports under a generated-runtime-like rextio.

Core's project build emits ``.rextio/build/python/rextio`` with only
``__about__``, ``__init__``, and ``runtime``. Putting that directory first on
``PYTHONPATH`` shadows a full install. Fallback wrappers still do
``from rextio_numpy.types import ...``; that path must not pull
``rextio.config`` (or analyzer/plugin host modules) at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _write_minimal_rextio(root: Path) -> Path:
    """Create a generated-runtime-shaped rextio package; return its parent path."""
    runtime_python = root / "build" / "python"
    rextio_pkg = runtime_python / "rextio"
    rextio_pkg.mkdir(parents=True)
    (rextio_pkg / "__init__.py").write_text(
        '"""Minimal generated-runtime rextio package."""\n',
        encoding="utf-8",
    )
    (rextio_pkg / "__about__.py").write_text(
        '__version__ = "0.0.0-generated-runtime"\n',
        encoding="utf-8",
    )
    (rextio_pkg / "runtime.py").write_text(
        '"""Generated runtime helpers stub."""\n',
        encoding="utf-8",
    )
    assert not (rextio_pkg / "config").exists()
    assert not (rextio_pkg / "plugins").exists()
    assert not (rextio_pkg / "analyzer").exists()
    return runtime_python


def test_types_and_root_exports_import_without_rextio_config(tmp_path: Path) -> None:
    """Shadow full core with minimal rextio; types + root exports must import."""
    runtime_python = _write_minimal_rextio(tmp_path)

    script = textwrap.dedent(
        """\
        import importlib.util
        import sys

        # Sanity: the shadowed package has no config/plugins/analyzer.
        assert importlib.util.find_spec("rextio") is not None
        assert importlib.util.find_spec("rextio.config") is None
        assert importlib.util.find_spec("rextio.plugins") is None
        assert importlib.util.find_spec("rextio.analyzer") is None

        import rextio

        assert not hasattr(rextio, "config")

        # Primary failure mode before the fix: package init -> plugin -> config.
        import rextio_numpy.types as types

        assert types.F64Arr1 is not None
        assert types.F64Arr2 is not None

        from rextio_numpy import RextioNumpyPlugin, __version__, plugin

        assert isinstance(__version__, str) and __version__
        assert callable(plugin)
        provider = plugin()
        assert isinstance(provider, RextioNumpyPlugin)
        assert provider.plugin_id == "rextio-numpy"
        assert provider.api_version == "1.2"
        print("ok")
        """
    )

    env = dict(os.environ)
    # Generated runtime first, then editable src — mirrors built-project layout.
    env["PYTHONPATH"] = os.pathsep.join((str(runtime_python), str(SRC_ROOT)))

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert "ok" in completed.stdout


def test_would_fail_if_plugin_eagerly_imported_config(tmp_path: Path) -> None:
    """Guard: importing rextio.config under the same shadow must still fail.

    Documents that the environment truly lacks config — the previous test is
    not a false green from an accidental full-core import.
    """
    runtime_python = _write_minimal_rextio(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(runtime_python), str(SRC_ROOT)))

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rextio.config  # noqa: F401\n",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode != 0
    assert "rextio.config" in (completed.stderr + completed.stdout)


def test_root_plugin_export_stable_after_entry_point_submodule_import() -> None:
    """Package-root ``plugin`` must stay the factory after submodule import.

    Importing ``rextio_numpy.plugin`` first binds the submodule name on the
    package. A lazy ``__getattr__`` root export can then return the *module*
    instead of the entry-point factory. Eager re-export binds the callable.

    Runs in a fresh subprocess so prior test imports cannot mask the bug.
    """
    script = textwrap.dedent(
        """\
        import importlib
        import sys
        import types

        # Isolate: drop any already-loaded package state (defensive; subprocess
        # should start clean, but keep the contract explicit).
        for name in list(sys.modules):
            if name == "rextio_numpy" or name.startswith("rextio_numpy."):
                del sys.modules[name]

        # Entry-point path first (core loader does importlib on this module).
        ep_mod = importlib.import_module("rextio_numpy.plugin")
        assert isinstance(ep_mod, types.ModuleType)
        assert callable(ep_mod.plugin)
        assert isinstance(ep_mod.RextioNumpyPlugin, type)

        # Root public exports must be the factory/class, not the submodule.
        import rextio_numpy

        assert rextio_numpy.plugin is ep_mod.plugin
        assert rextio_numpy.RextioNumpyPlugin is ep_mod.RextioNumpyPlugin
        assert callable(rextio_numpy.plugin)
        assert not isinstance(rextio_numpy.plugin, types.ModuleType)
        assert isinstance(rextio_numpy.RextioNumpyPlugin, type)

        from rextio_numpy import RextioNumpyPlugin, plugin

        assert plugin is ep_mod.plugin
        assert RextioNumpyPlugin is ep_mod.RextioNumpyPlugin
        assert callable(plugin)
        assert not isinstance(plugin, types.ModuleType)
        provider = plugin()
        assert isinstance(provider, RextioNumpyPlugin)
        assert provider.plugin_id == "rextio-numpy"
        assert provider.api_version == "1.2"
        print("ok")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (
                    [str(SRC_ROOT)]
                    + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
                )
            ),
        },
        check=False,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert "ok" in completed.stdout
