"""Top-level control: plan -> materialise -> run the 4 stages -> pass or escalate.

The orchestrator owns sequencing, checkpoint/rollback, and the ledger. It never
edits a dictionary or reads a log directly -- that is all stage/tool work.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from . import foam
from .agents import plan
from .llm import LLMClient
from .report import Reporter, make_reporter
from .schema import CaseSpec, CaseState, RunLedger, Stage, StageResult
from .stages import STAGES, execute_stage
from .tools import Workspace, dispatch

TEMPLATE_CASE = Path(__file__).resolve().parent.parent / "cases" / "cavity"
STAGE_ORDER = [Stage.PRE, Stage.SETUP, Stage.SOLVE, Stage.POST]


def run(
    request: str,
    *,
    model: str | None = None,
    root: str | Path = "runs/latest",
    report: Reporter | None = None,
) -> RunLedger:
    rep = report or make_reporter()
    llm = LLMClient(model=model) if model else LLMClient()
    ws = Workspace(root)
    ws.register("template", TEMPLATE_CASE)

    ledger = RunLedger(request=request, model=llm.model)
    rep.run_start(request, llm.model)

    spec = plan(request, llm)
    rep.plan(spec)

    ledger.add_case(spec)
    rep.case_start(spec)
    _materialise(ws, spec, rep)
    state = CaseState(spec=spec, path=str(ws.path(spec.name)), stage=Stage.PRE)

    ok, results = _run_stages(ws, state, llm, ledger, rep)
    if ok:
        return _finish(ledger, "pass", _summary(ledger, spec, results), rep)

    failed = results[-1]
    reason = f"{failed.stage.value} failed after {failed.attempt} attempt(s): {failed.gate}"
    return _finish(ledger, "escalated", reason, rep)


# --------------------------------------------------------------------------- #
def _run_stages(ws, state, llm, ledger, rep: Reporter) -> tuple[bool, list[StageResult]]:
    results: list[StageResult] = []
    prev: StageResult | None = None
    for stage in STAGE_ORDER:
        state.stage = stage
        snap = foam.snapshot(state.path, ws.snap_dir(state.spec.name), f"{stage.value}_entry")
        state.last_checkpoint = str(snap)
        try:
            r = execute_stage(STAGES[stage], ws, state, llm, prev, report=rep)
        except Exception as e:  # noqa: BLE001
            rep.crash(stage, f"{type(e).__name__}: {e}", traceback.format_exc())
            r = StageResult(case=state.spec.name, stage=stage, attempt=state.stage_attempts,
                            status="failed", gate=f"stage raised {type(e).__name__}: {e}")
        ledger.record(r)
        results.append(r)
        _record_recovery(r, ledger, rep)
        if not r.passed:
            foam.rollback(state.path, snap)
            rep.rollback(state.spec.name, stage)
            return False, results
        prev = r
    return True, results


def _materialise(ws: Workspace, spec: CaseSpec, rep: Reporter) -> None:
    """Create the case dir and apply the identity part of the spec (nu, solver
    application). Time controls + mesh are stage/LLM work."""
    res = dispatch(ws, "clone_case", {"source": "template", "dest": spec.name})
    if res.get("error"):
        raise RuntimeError(f"could not create case {spec.name}: {res['error']}")
    dispatch(ws, "set_config", {"case": spec.name, "file": "transportProperties", "key": "nu", "value": f"{spec.nu:g}"})
    dispatch(ws, "set_config", {"case": spec.name, "file": "controlDict", "key": "application", "value": spec.solver})
    rep.note(f"created {spec.name}: nu={spec.nu:g}, {spec.solver}")


def _record_recovery(r: StageResult, ledger: RunLedger, rep: Reporter) -> None:
    if r.stage == Stage.SOLVE and r.attempt > 1:
        v = r.metrics.get("verdict", "?")
        msg = f"SOLVE on {r.case}: converged after {r.attempt} attempts ({v})"
        ledger.recoveries.append(msg)
        if r.passed:
            rep.recovery(msg)
    if r.stage == Stage.PRE and r.attempt > 1 and r.passed:
        msg = f"PRE on {r.case}: mesh quality fixed over {r.attempt} attempts"
        ledger.recoveries.append(msg)
        rep.recovery(msg)


def _summary(ledger: RunLedger, spec: CaseSpec, results: list[StageResult]) -> str:
    v = results[-1].metrics  # POST: extract_primary_vortex output
    return (
        f"PASS. Final case {spec.name} ({spec.mesh.counts[0]}x{spec.mesh.counts[1]}). "
        f"Primary vortex at x/L={v.get('x_norm')}, y/L={v.get('y_norm')}. "
        f"Recoveries: {len(ledger.recoveries)}. "
        f"Tool calls: {sum(len(r.tool_calls) for r in ledger.stage_results)}."
    )


def _finish(ledger: RunLedger, verdict: str, summary: str, rep: Reporter) -> RunLedger:
    ledger.verdict = verdict
    ledger.summary = summary
    rep.run_end(ledger)
    return ledger
