"""Notebook helpers: pull, rebuild stale native deps, reload cached Python modules.

Use after ``git pull`` in Colab/Jupyter so you do not need a full runtime restart
(or unconditional CPRF/PRC rebuilds) to pick up repo changes.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

_CORE_RELOAD_ORDER: tuple[str, ...] = (
    "prc",
    "cprf",
    "text_attributes",
    "randrecover",
    "watermarking",
    "model",
    "app",
)

_BENCHMARK_RELOAD_ORDER: tuple[str, ...] = (
    "benchmark_io",
    "benchmark_plot",
    "benchmark_policy_detection",
    "benchmark_watermark",
    "benchmark_ber_diagnostics",
)

# Back-compat alias for callers that referenced the full list.
_RELOAD_ORDER: tuple[str, ...] = _CORE_RELOAD_ORDER + _BENCHMARK_RELOAD_ORDER

_CPRF_SOURCES = ("cprf.go",)
_PRC_SOURCE_GLOBS = ("src/**/*.rs", "Cargo.toml", "Cargo.lock", "pyproject.toml")


def _max_mtime(paths: Iterable[Path]) -> float:
    latest = 0.0
    for path in paths:
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
    return latest


def _collect_paths(root: Path, patterns: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        if "*" in pattern:
            out.extend(root.glob(pattern))
        else:
            candidate = root / pattern
            if candidate.is_file():
                out.append(candidate)
    return out


def git_pull(project_root: Path | str, *, ff_only: bool = True) -> str:
    """Run ``git pull`` and return the latest ``git log -1 --oneline`` line."""
    root = Path(project_root)
    if not (root / ".git").is_dir():
        raise FileNotFoundError(f"Not a git repo: {root}")
    flag = ["--ff-only"] if ff_only else []
    subprocess.run(["git", "-C", str(root), "pull", *flag], check=True)
    log = subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--oneline"],
        capture_output=True,
        text=True,
        check=True,
    )
    return log.stdout.strip()


def cprf_needs_rebuild(project_root: Path | str) -> bool:
    root = Path(project_root)
    cprf_dir = root / "cprf"
    so_path = cprf_dir / "cprf.so"
    if not so_path.is_file():
        return True
    src_mtime = _max_mtime(cprf_dir / name for name in _CPRF_SOURCES)
    return src_mtime > so_path.stat().st_mtime


def build_cprf(project_root: Path | str, *, go_exe: str = "go") -> Path:
    root = Path(project_root)
    cprf_dir = root / "cprf"
    so_path = cprf_dir / "cprf.so"
    subprocess.run(
        [go_exe, "build", "-o", "cprf.so", "-buildmode=c-shared", "cprf.go"],
        cwd=cprf_dir,
        check=True,
    )
    if not so_path.is_file():
        raise FileNotFoundError(f"CPRF build did not produce {so_path}")
    return so_path


def ensure_cprf(project_root: Path | str, *, go_exe: str = "go", force: bool = False) -> bool:
    """Build ``cprf.so`` only when missing or Go sources are newer. Returns True if rebuilt."""
    if force or cprf_needs_rebuild(project_root):
        build_cprf(project_root, go_exe=go_exe)
        return True
    return False


def prc_needs_rebuild(project_root: Path | str) -> bool:
    root = Path(project_root)
    prc_dir = root / "prc"
    wheel_dir = prc_dir / "target" / "wheels"
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        return True
    src_mtime = _max_mtime(_collect_paths(prc_dir, _PRC_SOURCE_GLOBS))
    return src_mtime > wheels[0].stat().st_mtime


def build_prc(project_root: Path | str, *, python: str | None = None) -> Path:
    root = Path(project_root)
    py = python or sys.executable
    subprocess.run([py, "-m", "pip", "install", "-q", "maturin"], check=True)
    cp = subprocess.run(
        ["maturin", "build", "--release", "-m", "prc/Cargo.toml"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        if cp.stdout:
            print(cp.stdout)
        if cp.stderr:
            print(cp.stderr, file=sys.stderr)
        cp.check_returncode()
    wheel_dir = root / "prc" / "target" / "wheels"
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        raise FileNotFoundError("No wheel in prc/target/wheels after maturin build")
    subprocess.run([py, "-m", "pip", "install", "-q", "--force-reinstall", str(wheels[0])], check=True)
    return wheels[0]


def ensure_prc(project_root: Path | str, *, python: str | None = None, force: bool = False) -> bool:
    """Build and pip-install PRC only when Rust sources are newer than the latest wheel."""
    if force or prc_needs_rebuild(project_root):
        build_prc(project_root, python=python)
        return True
    return False


def unload_model() -> None:
    try:
        import model

        model.unload()
    except Exception:
        pass


def reload_scheme(
    project_root: Path | str,
    *,
    unload_model_first: bool = True,
    include_benchmarks: bool = True,
    extra_modules: Sequence[str] = (),
) -> list[str]:
    """Evict cached scheme modules and import them fresh from ``project_root``.

    Call this after ``git pull`` (and any native rebuild) so notebook cells pick up
    changed ``.py`` files without a runtime restart. CPRF/PRC must be rebuilt
    separately when their native sources change (see ``ensure_cprf`` / ``ensure_prc``).
    """
    root = Path(project_root).resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root)

    if unload_model_first:
        unload_model()

    order = list(_CORE_RELOAD_ORDER)
    if include_benchmarks:
        order.extend(_BENCHMARK_RELOAD_ORDER)
    order.extend(m for m in extra_modules if m not in order)
    for name in order:
        sys.modules.pop(name, None)

    reloaded: list[str] = []
    skipped: list[str] = []
    for name in order:
        try:
            importlib.import_module(name)
            reloaded.append(name)
        except ImportError:
            skipped.append(name)
    if skipped:
        print("reload_scheme: skipped (missing deps):", ", ".join(skipped))
    return reloaded


def sync_after_pull(
    project_root: Path | str,
    *,
    go_exe: str = "go",
    python: str | None = None,
    force_native: bool = False,
) -> dict[str, object]:
    """``git pull`` → rebuild native deps only if stale → reload Python modules."""
    root = Path(project_root)
    head = git_pull(root)
    rebuilt_cprf = ensure_cprf(root, go_exe=go_exe, force=force_native)
    rebuilt_prc = ensure_prc(root, python=python, force=force_native)
    reloaded = reload_scheme(root)
    return {
        "head": head,
        "rebuilt_cprf": rebuilt_cprf,
        "rebuilt_prc": rebuilt_prc,
        "reloaded_modules": reloaded,
    }
