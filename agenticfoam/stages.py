"""The stage machine.

Each stage is a bounded LLM executor sub-loop:

    fresh context  ->  LLM emits tool calls  ->  dispatcher runs them  ->  repeat
                       until the LLM stops  ->  DETERMINISTIC GATE
                       gate pass -> advance ; gate fail -> feed the verdict
                       back and let the executor try again (up to max_attempts)

The LLM chooses *what* to change; the gate (gates.py) decides whether it worked.
Context is rebuilt per stage from the CaseSpec + CaseState + previous
StageResult -- it never grows into one long transcript.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from . import foam, gates
from .foam import time_dirs
from .llm import LLMClient, assistant_msg, sys_msg, tool_msg, user_msg
from .report import Reporter
from .schema import CaseState, Stage, StageResult, ToolCall
from .tools import SPEC, Workspace, dispatch, tool_schemas

MAX_TURNS_PER_ATTEMPT = 8
DEFAULT_MAX_ATTEMPTS = 3

EXECUTOR_SYSTEM = """\
You are the executor for one stage of an automated OpenFOAM lid-driven cavity workflow.

Rules:
- Act only through tool calls. One or a few at a time; inspect results before the next move.
- Before a tool call, write ONE short sentence saying what you are about to do and why.
- You cannot see the filesystem except through tools. Never assume a value you have not read.
- Stay within this stage's remit. Do not try to run later stages.
- When you believe this stage's exit gate will pass, STOP: reply with a one-sentence
  rationale and NO tool call. A deterministic gate then checks your work. If it fails,
  you get the verdict and try again.
- Diagnose from evidence in tool results (solver logs, checkMesh numbers).
"""


@dataclass
class StageDef:
    stage: Stage
    tools: list[str]
    max_attempts: int
    brief: Callable[[CaseState, StageResult | None], str]
    gate: Callable[[Workspace, CaseState], gates.Gate]


# --------------------------------------------------------------------------- #
# Gates wired to the workspace
# --------------------------------------------------------------------------- #
def _gate_pre(ws: Workspace, st: CaseState) -> gates.Gate:
    return gates.gate_pre(dispatch(ws, "mesh_quality", {"case": st.spec.name}))


def _gate_setup(ws: Workspace, st: CaseState) -> gates.Gate:
    case = ws.path(st.spec.name)
    for fld in ("0/U", "0/p"):
        if not (case / fld).is_file():
            return gates.Gate(False, f"missing initial field {fld}", {})
    # patch names referenced in 0/U must exist in the mesh
    bnd = case / "constant" / "polyMesh" / "boundary"
    if bnd.is_file():
        import re

        mesh_patches = set(re.findall(r"^\s*([A-Za-z]\w*)\s*\n\s*\{", bnd.read_text(), re.MULTILINE))
        u_patches = set(re.findall(r"^\s{4}([A-Za-z]\w*)\s*\n\s{4}\{", (case / "0" / "U").read_text(), re.MULTILINE))
        missing = u_patches - mesh_patches - {"boundaryField"}
        if missing:
            return gates.Gate(False, f"0/U references patches not in the mesh: {sorted(missing)}", {})

    # design doc: SETUP checks "the kinematic viscosity ... and time-control
    # parameters" -- not just that the field files exist
    s = st.spec
    for rel, key, want in (
        ("constant/transportProperties", "nu", s.nu),
        ("system/controlDict", "deltaT", s.delta_t),
        ("system/controlDict", "endTime", s.end_time),
    ):
        try:
            got = float(foam.get_entry(case / rel, key))
        except Exception as e:  # noqa: BLE001
            return gates.Gate(False, f"could not read {key} from {rel}: {e}", {})
        if abs(got - want) > max(1e-9, abs(want) * 1e-3):
            return gates.Gate(False, f"{rel}: {key} = {got:g}, expected {want:g}", {})

    return gates.Gate(True, "initial fields, viscosity, and time controls all consistent with the plan", {})


def _gate_solve(ws: Workspace, st: CaseState) -> gates.Gate:
    v, reason, m = gates.classify_convergence(ws.path(st.spec.name), st.spec.solver)
    return gates.gate_solve(v, reason, m)


def _gate_post(ws: Workspace, st: CaseState) -> gates.Gate:
    dispatch(ws, "run_postprocess", {"case": st.spec.name, "func": "writeCellCentres"})
    return gates.gate_post(dispatch(ws, "extract_primary_vortex", {"case": st.spec.name}))


# --------------------------------------------------------------------------- #
# Briefs
# --------------------------------------------------------------------------- #
def _brief_pre(st: CaseState, _prev: StageResult | None) -> str:
    m = st.spec.mesh
    return (
        f"STAGE: PRE (mesh) for case '{st.spec.name}', Re = {st.spec.reynolds:g}.\n"
        f"Target mesh: {m.counts[0]}x{m.counts[1]} cells, wall grading {m.grading[0]}x{m.grading[1]}.\n"
        f"Do: make system/blockMeshDict match that target (set_mesh), run blockMesh, and confirm.\n"
        f"Exit gate: blockMesh succeeds and checkMesh quality is in bounds "
        f"(non-orthogonality < {gates.NONORTHO_MAX:g}, aspect < {gates.ASPECT_MAX:g}, skewness < {gates.SKEW_MAX:g}).\n"
        f"If the gate fails on quality, relax the grading or add cells."
    )


def _brief_setup(st: CaseState, _prev: StageResult | None) -> str:
    s = st.spec
    lines = [
        f"STAGE: SETUP for case '{s.name}', Re = {s.reynolds:g}.",
        "Required physical state:",
        f"  - constant/transportProperties: nu = {s.nu:g}  (Re = U*L/nu, U=1, L=0.1)",
        f"  - system/controlDict: solver application = {s.solver}, deltaT = {s.delta_t:g}, "
        f"endTime = {s.end_time:g}, writeControl timeStep, writeInterval = {s.write_interval_steps}, "
        f"startFrom = startTime, startTime = 0.",
        "Read each key before you change it. Exit gate: initial fields present, patch names "
        "consistent with the mesh, and nu/deltaT/endTime read back matching the values above.",
    ]
    return "\n".join(lines)


def _brief_solve(st: CaseState, _prev: StageResult | None) -> str:
    s = st.spec
    return (
        f"STAGE: SOLVE for case '{s.name}', Re = {s.reynolds:g}, solver {s.solver}.\n"
        f"Do: run the solver. Then judge the run from its log.\n"
        f"Exit gate (deterministic classifier):\n"
        f"  - CONVERGED (U no longer iterating, and p's initial residual < 1e-6) -> pass.\n"
        f"  - NOT_YET (not there yet, but stable) -> extend controlDict endTime, set startFrom latestTime, run again.\n"
        f"  - DIVERGING (Courant > 1, residual blow-up) -> the timestep is too large. Read the max Courant "
        f"from the solver result and cut deltaT so the new max Courant is about 0.5 "
        f"(new deltaT = old deltaT * 0.5 / observed_max_Courant). If a diverged time dir was written, "
        f"you may need to reset startFrom/startTime to a clean state before rerunning.\n"
        f"Iterate until the classifier returns CONVERGED."
    )


def _brief_post(st: CaseState, _prev: StageResult | None) -> str:
    return (
        f"STAGE: POST for case '{st.spec.name}'.\n"
        f"Do: run postProcess writeCellCentres at the latest time, then extract the primary vortex location.\n"
        f"Exit gate: a primary-vortex (x, y) is successfully extracted."
    )


STAGES: dict[Stage, StageDef] = {
    Stage.PRE: StageDef(Stage.PRE, ["read_config", "set_config", "set_mesh", "run_blockmesh", "mesh_quality", "read_log"], 3, _brief_pre, _gate_pre),
    Stage.SETUP: StageDef(Stage.SETUP, ["read_config", "set_config", "list_times", "read_log"], 3, _brief_setup, _gate_setup),
    Stage.SOLVE: StageDef(Stage.SOLVE, ["read_config", "set_config", "run_solver", "read_log", "list_times"], 5, _brief_solve, _gate_solve),
    Stage.POST: StageDef(Stage.POST, ["run_postprocess", "extract_primary_vortex", "list_times"], 2, _brief_post, _gate_post),
}


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
def _schemas_for(names: list[str]) -> list[dict]:
    allow = set(names)
    return [s for s in tool_schemas() if s["function"]["name"] in allow]


def _refresh(ws: Workspace, st: CaseState) -> None:
    st.time_dirs = time_dirs(ws.path(st.spec.name))


def execute_stage(
    sd: StageDef,
    ws: Workspace,
    st: CaseState,
    llm: LLMClient,
    prev: StageResult | None,
    *,
    report: "Reporter",
) -> StageResult:
    t0 = time.time()
    schemas = _schemas_for(sd.tools)
    transcript = [sys_msg(EXECUTOR_SYSTEM), user_msg(sd.brief(st, prev))]
    trace: list[ToolCall] = []
    last_rationale = ""
    gate = gates.Gate(False, "stage did not run", {})

    for attempt in range(1, sd.max_attempts + 1):
        st.stage_attempts = attempt
        report.stage_start(sd.stage, attempt, sd.max_attempts)

        for _turn in range(MAX_TURNS_PER_ATTEMPT):
            reply = llm.chat(transcript, tools=schemas)
            transcript.append(assistant_msg(reply))
            if reply.text:
                last_rationale = reply.text.strip()
                report.rationale(reply.text)
            if reply.finish_reason == "length" and not reply.wants_tools:
                transcript.append(user_msg(
                    "Your reply was cut off. Be concise: make exactly one tool call, "
                    "or give a one-sentence rationale with no tool call."
                ))
                continue
            if not reply.wants_tools:
                break
            for tc in reply.tool_calls:
                if tc.name not in SPEC:
                    transcript.append(tool_msg(tc.id, {"error": f"tool '{tc.name}' not available in this stage"}))
                    continue
                report.tool(tc.name, tc.arguments)
                out = dispatch(ws, tc.name, tc.arguments)
                ok = not (isinstance(out, dict) and out.get("error"))
                trace.append(ToolCall(name=tc.name, args=tc.arguments, ok=ok, result=_trim(out)))
                report.tool_result(tc.name, tc.arguments, ok, out)
                transcript.append(tool_msg(tc.id, out))
            _refresh(ws, st)

        gate = sd.gate(ws, st)
        report.gate(sd.stage, gate.passed, gate.reason)
        if gate.passed:
            break
        transcript.append(user_msg(
            f"GATE FAILED: {gate.reason}\n"
            f"Diagnose from the evidence, apply one fix, then stop when you expect the gate to pass."
        ))

    _refresh(ws, st)
    return StageResult(
        case=st.spec.name,
        stage=sd.stage,
        attempt=st.stage_attempts,
        status="passed" if gate.passed else "failed",
        gate=gate.reason,
        metrics=gate.metrics,
        artifacts=[str(p.relative_to(ws.root)) for p in ws.path(st.spec.name).glob("log.*")],
        tool_calls=trace,
        rationale=last_rationale,
        elapsed_s=round(time.time() - t0, 1),
    )


# --------------------------------------------------------------------------- #
def _trim(d, n: int = 600):
    """Clip long values before they go into the stored ToolCall trace."""
    s = d if isinstance(d, dict) else {"value": d}
    return {k: (str(v)[:n] + "..." if len(str(v)) > n else v) for k, v in s.items()}
