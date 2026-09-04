"""Low-level OpenFOAM layer: run a solver/utility, parse its log, enumerate time
directories, snapshot / roll back a case.

Nothing in here knows about the LLM. Everything the agent does to a case
ultimately goes through these primitives.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# The OpenFOAM environment is sourced per-call, exactly like the ./foam wrapper
# used during the manual walkthrough. Override for a different install.
BASHRC = os.environ.get(
    "AGENTICFOAM_BASHRC", "/usr/lib/openfoam/openfoam2412/etc/bashrc"
)

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


@dataclass
class FoamRun:
    """Result of running one OpenFOAM binary."""

    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    wall_seconds: float
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.fatal_error is None

    @property
    def fatal_error(self) -> str | None:
        return parse_fatal_error(self.stdout + "\n" + self.stderr)


def run_foam(
    argv: list[str],
    cwd: str | Path,
    *,
    timeout: float = 600.0,
    log_name: str | None = None,
) -> FoamRun:
    """Run ``argv`` (e.g. ``["icoFoam"]`` or ``["blockMesh", "-dict", ...]``) inside
    an OpenFOAM environment, in directory ``cwd``.

    If ``log_name`` is given, combined stdout is also written to ``cwd/log_name``
    (matching OpenFOAM's own ``log.<app>`` convention).
    """
    cwd = Path(cwd)
    # `source bashrc; exec "$@"` — bashrc is noisy on stdout, so silence it.
    wrapped = ["bash", "-c", f'source "{BASHRC}" >/dev/null 2>&1; exec "$@"', "_", *argv]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            wrapped,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exit_code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        exit_code = 124
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (
            (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        ) + f"\n[agenticfoam] killed after {timeout:.0f}s timeout"

    wall = time.monotonic() - t0
    log_path = None
    if log_name:
        log_path = cwd / log_name
        log_path.write_text(out + ("\n" + err if err else ""))
    return FoamRun(list(argv), exit_code, out, err, wall, log_path)


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #
@dataclass
class SolverLog:
    """Structured view of an icoFoam / pisoFoam log."""

    times: list[float] = field(default_factory=list)
    co_mean: list[float] = field(default_factory=list)
    co_max: list[float] = field(default_factory=list)
    # initial residual per field, indexed parallel to `times` where available
    residual_initial: dict[str, list[float]] = field(default_factory=dict)
    # solver iteration count per field (the "No Iterations N" figure)
    iterations: dict[str, list[float]] = field(default_factory=dict)
    diverged: bool = False
    completed: bool = False

    @property
    def max_courant(self) -> float:
        return max(self.co_max, default=0.0)

    @property
    def last_time(self) -> float:
        return self.times[-1] if self.times else 0.0


def parse_solver_log(text: str) -> SolverLog:
    log = SolverLog()
    cur_time: float | None = None
    for line in text.splitlines():
        m = re.match(r"^Time = (" + _NUM + r")\b", line)
        if m:
            cur_time = float(m.group(1))
            log.times.append(cur_time)
            continue
        m = re.search(
            r"Courant Number mean: (" + _NUM + r") max: (" + _NUM + r")", line
        )
        if m:
            log.co_mean.append(float(m.group(1)))
            log.co_max.append(float(m.group(2)))
            continue
        m = re.search(
            r"Solving for (\w+), Initial residual = (" + _NUM + r"), "
            r"Final residual = (" + _NUM + r"), No Iterations (\d+)",
            line,
        )
        if m:
            fld = m.group(1)
            log.residual_initial.setdefault(fld, []).append(float(m.group(2)))
            log.iterations.setdefault(fld, []).append(float(m.group(4)))
            continue
        if re.search(r"\b(nan|inf)\b", line, re.IGNORECASE) and "Solving for" in line:
            log.diverged = True
    if re.search(r"^End\b", text, re.MULTILINE):
        log.completed = True
    # A runaway solution shows up as exploding initial residuals too.
    for series in log.residual_initial.values():
        if series and (series[-1] > 1e6 or _has_nan(series)):
            log.diverged = True
    return log


def _has_nan(xs: list[float]) -> bool:
    return any(x != x for x in xs)


def parse_fatal_error(text: str) -> str | None:
    """Return the OpenFOAM ``--> FOAM FATAL (IO )?ERROR`` block, if present."""
    m = re.search(r"--> FOAM FATAL (?:IO )?ERROR.*", text, re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    # Trim to the first blank line after the "From ..." / "in file ..." trailer,
    # or ~1500 chars, whichever comes first.
    cut = re.search(r"\n\s*\n", block)
    block = block[: cut.start()] if cut else block
    return block[:1500].strip()


# --------------------------------------------------------------------------- #
# Case filesystem
# --------------------------------------------------------------------------- #
def time_dirs(case: str | Path) -> list[str]:
    """Numeric time directories present in a case, sorted ascending, as strings
    (so '0.005' etc. round-trip exactly)."""
    case = Path(case)
    out: list[tuple[float, str]] = []
    for p in case.iterdir():
        if not p.is_dir():
            continue
        try:
            out.append((float(p.name), p.name))
        except ValueError:
            continue
    return [name for _, name in sorted(out)]


def latest_time(case: str | Path) -> str | None:
    tds = time_dirs(case)
    return tds[-1] if tds else None


def snapshot(case: str | Path, dest_dir: str | Path, label: str) -> Path:
    """Copy the whole case tree to ``dest_dir/label`` and return that path."""
    case, dest_dir = Path(case), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / label
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(case, dst, symlinks=True)
    return dst


def rollback(case: str | Path, snap: str | Path) -> None:
    """Restore ``case`` to the state captured in ``snap``."""
    case, snap = Path(case), Path(snap)
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(snap, case, symlinks=True)


# --------------------------------------------------------------------------- #
# Dictionary editing -- a targeted line editor (not a parser) for the handful
# of keywords the harness touches: nu, deltaT, endTime, startFrom, and the
# single `blocks` line of blockMeshDict.
# --------------------------------------------------------------------------- #
class DictError(RuntimeError):
    pass


def get_entry(path: str | Path, key: str) -> str:
    """Value of a ``key   value;`` entry, verbatim."""
    m = re.search(rf"^\s*{re.escape(key)}\s+(.+?);\s*$", Path(path).read_text(), re.MULTILINE)
    if not m:
        raise DictError(f"{Path(path).name}: entry '{key}' not found")
    return m.group(1).strip()


def set_entry(path: str | Path, key: str, value: str) -> None:
    """Set an existing ``key   value;`` entry (raises if the key is absent -- the
    harness only ever *changes* known keys)."""
    p = Path(path)
    pat = re.compile(rf"^(?P<i>\s*){re.escape(key)}\s+.+?;\s*$", re.MULTILINE)
    text, n = pat.subn(lambda m: f"{m['i']}{key}{' ' * max(1, 15 - len(key))}{value};", p.read_text())
    if n != 1:
        raise DictError(f"{p.name}: entry '{key}' matched {n} times, expected 1")
    p.write_text(text)


def mesh_spec(path: str | Path) -> dict:
    """{counts:[nx,ny,nz], grading:[gx,gy,gz]} for a one-block blockMeshDict."""
    m = re.search(
        r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*"
        r"simpleGrading\s*\(\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\)",
        Path(path).read_text(),
    )
    if not m:
        raise DictError(f"{Path(path).name}: no single-block mesh line")
    return {"counts": [int(m[1]), int(m[2]), int(m[3])],
            "grading": [float(m[4]), float(m[5]), float(m[6])]}


def set_mesh(
    path: str | Path,
    counts: tuple[int, int, int] | None = None,
    grading: tuple[float, float, float] | None = None,
) -> None:
    """Edit the single ``hex (...) (nx ny nz) simpleGrading (gx gy gz)`` line."""
    p = Path(path)
    text = p.read_text()
    pat = re.compile(
        r"hex\s*\(([^)]*)\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*"
        r"simpleGrading\s*\(\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\)"
    )
    m = pat.search(text)
    if not m:
        raise DictError(f"{p.name}: no single-block 'hex ... simpleGrading' line")
    nx, ny, nz = counts or (int(m[2]), int(m[3]), int(m[4]))
    gx, gy, gz = grading or (m[5], m[6], m[7])
    p.write_text(text[: m.start()] + f"hex ({m[1].strip()}) ({nx} {ny} {nz}) "
                 f"simpleGrading ({gx} {gy} {gz})" + text[m.end():])
