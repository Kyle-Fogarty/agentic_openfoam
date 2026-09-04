"""The planner LLM role that sits outside the stage machine:

  plan() -- free-text request -> CaseSpec (fills gaps, records assumptions)

The main per-stage executor lives in stages.py; this is the entry bookend.
"""
from __future__ import annotations

from .llm import LLMClient, extract_json, sys_msg, user_msg
from .schema import CaseSpec, MeshSpec, nu_for_reynolds

# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
_PLANNER_SYSTEM = """\
You convert a natural-language request into a plan for an automated OpenFOAM
lid-driven cavity run. Extract only what the user actually specified. Reply with
ONLY a JSON object using these keys (all optional):

  {"reynolds": <number>,
   "mesh_n": <int, cells per side for a uniform mesh>,
   "grading": <number, wall grading ratio, 1 = uniform>,
   "solver": "icoFoam" | "pisoFoam",
   "quantities": [<strings: what the user wants to know>],
   "notes": [<strings: anything ambiguous you had to interpret>]}

Do not invent an endTime or a timestep. Do not add keys.
"""

_PLAN_DEFAULTS = dict(mesh_n=20, grading=1.0, delta_t=0.005, end_time=0.5, write_interval_steps=20)


def plan(request: str, llm: LLMClient) -> CaseSpec:
    try:
        p = extract_json(llm.chat([sys_msg(_PLANNER_SYSTEM), user_msg(request)]).text)
    except Exception:  # noqa: BLE001
        p = {}

    assume: list[str] = list(p.get("notes") or [])

    re_num = p.get("reynolds")
    if not isinstance(re_num, (int, float)) or re_num <= 0:
        re_num = 100.0
        assume.append("Reynolds number not given -> assumed Re = 100")

    n = int(p.get("mesh_n") or _PLAN_DEFAULTS["mesh_n"])
    if "mesh_n" not in p:
        assume.append(f"mesh not specified -> starting at {n}x{n} uniform")
    grading = float(p.get("grading") or _PLAN_DEFAULTS["grading"])

    solver = p.get("solver") or ("pisoFoam" if re_num >= 1000 else "icoFoam")
    if "solver" not in p:
        assume.append(f"solver not specified -> {solver}")

    assume.append(
        f"exploratory endTime {_PLAN_DEFAULTS['end_time']} s / deltaT {_PLAN_DEFAULTS['delta_t']} s "
        "(template values); SOLVE gate extends endTime to convergence and cuts deltaT if unstable"
    )

    return CaseSpec(
        name=f"cavity_re{int(re_num)}",
        reynolds=float(re_num),
        nu=nu_for_reynolds(re_num),
        mesh=MeshSpec(counts=(n, n, 1), grading=(grading, grading, 1.0)),
        solver=solver,
        end_time=_PLAN_DEFAULTS["end_time"],
        delta_t=_PLAN_DEFAULTS["delta_t"],
        write_interval_steps=_PLAN_DEFAULTS["write_interval_steps"],
        quantities=p.get("quantities") or ["primary vortex location", "centreline u-profile"],
        assumptions=assume,
    )
