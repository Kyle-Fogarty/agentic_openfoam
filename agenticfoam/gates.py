"""Stage gates -- the deterministic checks that decide whether a stage may
advance. Cheap and mechanical: exit codes, parsed metrics. No LLM judgement is
involved -- a stage transition is authorized by machine-verifiable evidence,
never by the model deciding an output "looks" successful.

The convergence classifier follows the design doc exactly: stability first
(Courant < 1), then convergence -- the velocity solve no longer requires
iterations *and* the pressure-Poisson initial residual r_p is below its solver
tolerance (fvSolution: solvers.p.tolerance = 1e-6). Anything stable but not yet
meeting that bar is "unfinished", not unstable: extend endTime and continue.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .foam import parse_solver_log

# tuning knobs
CO_CEILING = 1.0
P_RESID_CONVERGED = 1e-6     # pressure-Poisson initial residual tolerance (fvSolution: solvers.p.tolerance);
                             # with U no longer iterating, this is the design doc's convergence criterion
NONORTHO_MAX = 70.0
ASPECT_MAX = 100.0
SKEW_MAX = 4.0


class Convergence(str, Enum):
    CONVERGED = "converged"        # advance
    NOT_YET = "not_yet"           # extend endTime + continue
    DIVERGING = "diverging"        # roll back, fix numerics


@dataclass
class Gate:
    passed: bool
    reason: str
    metrics: dict


def classify_convergence(case: str | Path, solver: str) -> tuple[Convergence, str, dict]:
    log_path = Path(case) / f"log.{solver}"
    if not log_path.is_file():
        return Convergence.DIVERGING, "no solver log produced", {}
    log = parse_solver_log(log_path.read_text())

    u_series = (
        log.residual_initial.get("Ux")
        or log.residual_initial.get("U")
        or []
    )
    u_iters = log.iterations.get("Ux") or log.iterations.get("U") or []
    p_series = log.residual_initial.get("p") or []
    last = u_series[-1] if u_series else None
    p_last = p_series[-1] if p_series else None
    metrics = {
        "steps": len(log.times),
        "last_time": log.last_time,
        "max_courant": round(log.max_courant, 4),
        "u_initial_residual_last": last,
        "p_initial_residual_last": p_last,
        "u_iters_tail": u_iters[-5:],
        "completed": log.completed,
    }

    if log.diverged or log.max_courant > CO_CEILING:
        return Convergence.DIVERGING, f"max Courant {log.max_courant:.2f} (limit {CO_CEILING:g}), residual blow-up={log.diverged}", metrics
    if not log.completed:
        return Convergence.DIVERGING, "solver stopped before endTime without reaching a solution", metrics

    iters_quiet = bool(u_iters) and all(k == 0 for k in u_iters[-3:])
    u_txt = f"U residual {last:.1e}" if last is not None else "no U residual in the log yet"
    p_txt = f"p residual {p_last:.1e}" if p_last is not None else "no p residual in the log yet"

    # design doc's criterion: velocity solve no longer requires iterations, and
    # the pressure-Poisson initial residual is below its solver tolerance
    if iters_quiet and p_last is not None and p_last < P_RESID_CONVERGED:
        return Convergence.CONVERGED, f"p residual {p_last:.1e} < {P_RESID_CONVERGED:.0e}, solver no longer iterating U", metrics

    return Convergence.NOT_YET, f"not yet converged ({u_txt}, {p_txt}); needs a longer endTime", metrics


# --------------------------------------------------------------------------- #
# Per-stage gates
# --------------------------------------------------------------------------- #
def gate_pre(mesh_metrics: dict) -> Gate:
    if mesh_metrics.get("error"):
        return Gate(False, f"checkMesh failed: {mesh_metrics['error']}", mesh_metrics)
    if mesh_metrics.get("failed_checks"):
        return Gate(False, "checkMesh reported failed checks (***)", mesh_metrics)
    ar = mesh_metrics.get("max_aspect_ratio") or 0
    no = mesh_metrics.get("max_non_orthogonality") or 0
    sk = mesh_metrics.get("max_skewness") or 0
    bad = []
    if ar > ASPECT_MAX:
        bad.append(f"aspect ratio {ar:.1f} > {ASPECT_MAX}")
    if no > NONORTHO_MAX:
        bad.append(f"non-orthogonality {no:.1f} > {NONORTHO_MAX}")
    if sk > SKEW_MAX:
        bad.append(f"skewness {sk:.2f} > {SKEW_MAX}")
    if bad:
        return Gate(False, "mesh quality out of bounds: " + "; ".join(bad), mesh_metrics)
    return Gate(True, f"mesh OK · {mesh_metrics.get('n_cells')} cells · "
                      f"non-orthogonality {no:.0f}° · aspect {ar:.1f}", mesh_metrics)


def gate_solve(verdict: Convergence, reason: str, metrics: dict) -> Gate:
    ok = verdict == Convergence.CONVERGED
    return Gate(ok, f"{verdict.value}: {reason}", metrics | {"verdict": verdict.value})


def gate_post(extract: dict) -> Gate:
    if extract.get("error"):
        return Gate(False, f"could not extract quantities: {extract['error']}", extract)
    if "x" not in extract:
        return Gate(False, "vortex extraction returned nothing usable", extract)
    return Gate(True, f"primary vortex at ({extract['x']:.4f}, {extract['y']:.4f}) m", extract)
