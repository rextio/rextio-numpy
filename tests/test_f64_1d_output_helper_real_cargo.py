"""Real-Cargo negative coverage for the private F64_1D output helper."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from rextio_numpy.rust_snippets.array_repr import (
    f64_1d_output_helper,
    f64_1d_return_helper,
)


@pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="F64_1D output-helper test requires cargo",
)
def test_partial_fill_error_cannot_expose_output_and_success_releases_borrow(
    tmp_path: Path,
) -> None:
    """An error returns no array; a completed fill can release to its owner."""
    numpy = pytest.importorskip("numpy")
    crate = tmp_path / "f64-output-helper"
    source_dir = crate / "src"
    source_dir.mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        textwrap.dedent(
            """\
            [package]
            name = "rextio_numpy_f64_output_helper_test"
            version = "0.0.0"
            edition = "2021"

            [dependencies]
            numpy = "=0.29.0"
            pyo3 = { version = "=0.29.0", features = ["auto-initialize"] }
            """
        ),
        encoding="utf-8",
    )
    source = "\n\n".join(
        (
            "use pyo3::prelude::*;",
            f64_1d_output_helper(),
            f64_1d_return_helper(),
            textwrap.dedent(
                """\
                fn main() {
                    Python::attach(|py| {
                        let failed = __rxtnp_f64_1d_output(py, 4, |out| {
                            out[0] = 99.0;
                            Err(pyo3::exceptions::PyValueError::new_err(
                                "injected partial fill failure",
                            ))
                        });
                        assert!(failed.is_err());

                        let frozen = __rxtnp_f64_1d_output(py, 4, |out| {
                            for (index, value) in out.iter_mut().enumerate() {
                                *value = index as f64 + 0.5;
                            }
                            Ok(())
                        })
                        .expect("complete fill must succeed");
                        assert_eq!(
                            frozen.as_slice().expect("fresh output is contiguous"),
                            &[0.5, 1.5, 2.5, 3.5],
                        );
                        let owner = __rxtnp_release_f64_1d(frozen)
                            .expect("readonly borrow must release");
                        assert_eq!(owner.len().expect("array length must be readable"), 4);
                        assert!(numpy::PyArrayMethods::try_readwrite(&owner).is_ok());
                    });
                }
                """
            ),
        )
    )
    (source_dir / "main.rs").write_text(source, encoding="utf-8")

    numpy_site_packages = str(Path(numpy.__file__).resolve().parent.parent)
    python_path = os.pathsep.join(
        part
        for part in (numpy_site_packages, os.environ.get("PYTHONPATH", ""))
        if part
    )
    completed = subprocess.run(
        ["cargo", "run", "--quiet"],
        cwd=crate,
        env={
            **os.environ,
            "PYO3_PYTHON": sys.executable,
            "PYTHONPATH": python_path,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
