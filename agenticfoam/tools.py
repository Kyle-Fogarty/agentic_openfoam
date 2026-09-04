"""The action space: the only things that touch the filesystem or run OpenFOAM.

The LLM never runs a solver -- it emits ``{"name": "run_solver", "arguments":
{...}}`` and the dispatcher below executes the matching function and returns a
plain dict. Tools never raise into the loop; failures come back as
``{"error": "..."}`` so the executor can read and react to them.

``tool_schemas()`` builds the OpenAI-style tool list straight from these
function signatures + the ``SPEC`` table, so the advertised schema and the real
implementation can never drift apart.
"""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Callable

from . import foam
from .analysis import primary_vortex

# --------------------------------------------------------------------------- #
# Workspace: resolves the case *names* the LLM uses to real directories.
# --------------------------------------------------------------------------- #
class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()   # absolute -- foam utilities run in varied cwds
        (self.root / "cases").mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        self.cases: dict[str, Path] = {}

    def register(self, name: str, path: str | Path) -> Path:
        self.cases[name] = Path(path)
        return self.cases[name]

    def path(self, name: str) -> Path:
        if name not in self.cases:
            raise KeyError(f"unknown case '{name}' (known: {sorted(self.cases)})")
        return self.cases[name]

    def snap_dir(self, name: str) -> Path:
        return self.root / "snapshots" / name


_FILE_ALIASES = {
    "controlDict": "system/controlDict",
    "transportProperties": "constant/transportProperties",
    "blockMeshDict": "system/blockMeshDict",
    "fvSolution": "system/fvSolution",
    "fvSchemes": "system/fvSchemes",
    "U": "0/U",
    "p": "0/p",
}


def _resolve_file(case_dir: Path, name: str) -> Path:
    return case_dir / _FILE_ALIASES.get(name, name)


# --------------------------------------------------------------------------- #
# Inspect
# --------------------------------------------------------------------------- #
def read_config(ws: Workspace, case: str, file: str, key: str) -> dict:
    """Read one keyword entry from an OpenFOAM dictionary file."""
    try:
        return {"key": key, "value": foam.get_entry(_resolve_file(ws.path(case), file), key)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def list_times(ws: Workspace, case: str) -> dict:
    """List the numeric time directories written for a case."""
    tds = foam.time_dirs(ws.path(case))
    return {"times": tds, "latest": tds[-1] if tds else None}


def read_log(ws: Workspace, case: str, name: str, tail_lines: int = 40) -> dict:
    """Return the last ``tail_lines`` lines of a solver/utility log file."""
    p = ws.path(case) / name
    if not p.is_file():
        return {"error": f"no log '{name}' in {case}"}
    lines = p.read_text().splitlines()
    return {"log": name, "tail": "\n".join(lines[-tail_lines:])}


def mesh_quality(ws: Workspace, case: str) -> dict:
    """Run checkMesh and return the key quality metrics."""
    r = foam.run_foam(["checkMesh", "-constant"], ws.path(case), log_name="log.checkMesh")
    out = r.stdout
    import re

    def grab(pat: str) -> float | None:
        m = re.search(pat, out)
        return float(m.group(1)) if m else None

    metrics = {
        "n_cells": int(grab(r"cells:\s+(\d+)") or 0) or None,
        "max_aspect_ratio": grab(r"Max aspect ratio = (" + foam._NUM + r")"),
        "max_non_orthogonality": grab(r"Max non-orthogonality = (" + foam._NUM + r")"),
        "max_skewness": grab(r"Max skewness = (" + foam._NUM + r")"),
        "mesh_ok": "Mesh OK." in out,
        "failed_checks": "***" in out,
    }
    return metrics


# --------------------------------------------------------------------------- #
# Mutate
# --------------------------------------------------------------------------- #
def set_config(ws: Workspace, case: str, file: str, key: str, value: str) -> dict:
    """Set one keyword entry in an OpenFOAM dictionary file (key must already exist)."""
    try:
        path = _resolve_file(ws.path(case), file)
        old = None
        try:
            old = foam.get_entry(path, key)
        except Exception:  # noqa: BLE001
            pass
        foam.set_entry(path, key, str(value))
        return {"file": file, "key": key, "old": old, "new": str(value)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def set_mesh(
    ws: Workspace,
    case: str,
    nx: int | None = None,
    ny: int | None = None,
    grading_x: float | None = None,
    grading_y: float | None = None,
) -> dict:
    """Edit the single-block blockMeshDict: cell counts and/or wall grading ratios."""
    try:
        path = _resolve_file(ws.path(case), "blockMeshDict")
        cur = foam.mesh_spec(path)
        counts = (
            nx if nx is not None else cur["counts"][0],
            ny if ny is not None else cur["counts"][1],
            1,
        )
        grading = (
            grading_x if grading_x is not None else cur["grading"][0],
            grading_y if grading_y is not None else cur["grading"][1],
            1.0,
        )
        foam.set_mesh(path, counts=counts, grading=grading)
        return {"old": cur, "new": foam.mesh_spec(path)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def clone_case(ws: Workspace, source: str, dest: str) -> dict:
    """Create a new case from ``source``'s constant/ + system/ + 0/."""
    try:
        src = ws.path(source)
        dst = ws.root / "cases" / dest
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for sub in ("constant", "system"):
            shutil.copytree(src / sub, dst / sub, symlinks=True)
        if (src / "0").is_dir():
            shutil.copytree(src / "0", dst / "0", symlinks=True)
        ws.register(dest, dst)
        return {"dest": dest, "path": str(dst)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def run_blockmesh(ws: Workspace, case: str) -> dict:
    """Generate the mesh from system/blockMeshDict."""
    r = foam.run_foam(["blockMesh"], ws.path(case), log_name="log.blockMesh")
    import re

    m = re.search(r"nCells:\s*(\d+)", r.stdout) or re.search(r"cells:\s+(\d+)", r.stdout)
    return {
        "ok": r.ok,
        "exit_code": r.exit_code,
        "n_cells": int(m.group(1)) if m else None,
        "fatal": r.fatal_error,
        "log": "log.blockMesh",
    }


def run_solver(ws: Workspace, case: str, solver: str = "icoFoam") -> dict:
    """Run the transient solver to endTime. Returns convergence-relevant numbers
    from the log: max Courant, whether it completed or diverged, the final U
    initial-residual, and the tail of the U iteration count."""
    r = foam.run_foam([solver], ws.path(case), log_name=f"log.{solver}", timeout=900)
    log = foam.parse_solver_log(r.stdout)
    u_res = None
    for f in ("Ux", "U", "Uy"):
        if log.residual_initial.get(f):
            u_res = log.residual_initial[f][-1]
            break
    return {
        "ok": r.ok and not log.diverged,
        "exit_code": r.exit_code,
        "completed": log.completed,
        "diverged": log.diverged,
        "max_courant": round(log.max_courant, 4),
        "last_time": log.last_time,
        "u_initial_residual": u_res,
        "u_iterations_tail": log.iterations.get("Ux", log.iterations.get("U", []))[-5:],
        "fatal": r.fatal_error,
        "log": f"log.{solver}",
    }


def run_postprocess(ws: Workspace, case: str, func: str = "writeCellCentres") -> dict:
    """Run a postProcess function object (e.g. writeCellCentres) at the latest time."""
    r = foam.run_foam(
        ["postProcess", "-func", func, "-latestTime"], ws.path(case),
        log_name=f"log.postProcess.{func}", timeout=300,
    )
    lt = foam.latest_time(ws.path(case))
    produced = []
    if lt:
        for name in ("C", "Cx", "Cy", "Cz"):
            if (ws.path(case) / lt / name).is_file():
                produced.append(f"{lt}/{name}")
    return {"ok": r.ok, "produced": produced, "fatal": r.fatal_error}


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def extract_primary_vortex(ws: Workspace, case: str, time: str | None = None) -> dict:
    """Locate the primary vortex centre (argmax |streamfunction|) at a given time
    (default latest). Requires cell centres -- run postProcess writeCellCentres first."""
    try:
        return primary_vortex(ws.path(case), time)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
# Registry + schema generation + dispatch
# --------------------------------------------------------------------------- #
SPEC: dict[str, dict[str, Any]] = {
    "read_config": {"fn": read_config, "group": "inspect",
                    "params": {"case": "case name", "file": "dict file: controlDict | transportProperties | blockMeshDict | fvSolution",
                               "key": "keyword to read"}},
    "list_times": {"fn": list_times, "group": "inspect",
                   "params": {"case": "case name"}},
    "read_log": {"fn": read_log, "group": "inspect",
                 "params": {"case": "case name", "name": "log file name, e.g. log.icoFoam",
                            "tail_lines": "how many trailing lines to return"}},
    "mesh_quality": {"fn": mesh_quality, "group": "inspect",
                     "params": {"case": "case name"}},
    "set_config": {"fn": set_config, "group": "mutate",
                   "params": {"case": "case name", "file": "dict file", "key": "keyword", "value": "new value (string)"}},
    "set_mesh": {"fn": set_mesh, "group": "mutate",
                 "params": {"case": "case name", "nx": "cells in x (optional)", "ny": "cells in y (optional)",
                            "grading_x": "x wall grading ratio (optional)", "grading_y": "y wall grading ratio (optional)"}},
    "clone_case": {"fn": clone_case, "group": "mutate",
                   "params": {"source": "case to copy from", "dest": "new case name"}},
    "run_blockmesh": {"fn": run_blockmesh, "group": "execute",
                      "params": {"case": "case name"}},
    "run_solver": {"fn": run_solver, "group": "execute",
                   "params": {"case": "case name", "solver": "icoFoam | pisoFoam"}},
    "run_postprocess": {"fn": run_postprocess, "group": "execute",
                        "params": {"case": "case name", "func": "function object, e.g. writeCellCentres"}},
    "extract_primary_vortex": {"fn": extract_primary_vortex, "group": "evaluate",
                               "params": {"case": "case name", "time": "time dir (optional, default latest)"}},
}

_DESCRIPTIONS = {name: (spec["fn"].__doc__ or "").strip() for name, spec in SPEC.items()}


def _json_type(annotation: Any) -> str:
    ann = str(annotation)
    if "int" in ann:
        return "integer"
    if "float" in ann:
        return "number"
    if "bool" in ann:
        return "boolean"
    return "string"


def tool_schemas() -> list[dict]:
    """OpenAI-style tool list, derived from the function signatures."""
    out = []
    for name, spec in SPEC.items():
        fn: Callable = spec["fn"]
        sig = inspect.signature(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            if pname == "ws":
                continue
            props[pname] = {
                "type": _json_type(p.annotation),
                "description": spec["params"].get(pname, ""),
            }
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": _DESCRIPTIONS[name],
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out


def dispatch(ws: Workspace, name: str, args: dict[str, Any]) -> dict:
    """Execute a tool call. Never raises: unknown tool / bad args / tool failure
    all come back as a dict the executor can read."""
    if name not in SPEC:
        return {"error": f"unknown tool '{name}'"}
    fn = SPEC[name]["fn"]
    try:
        return fn(ws, **args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
