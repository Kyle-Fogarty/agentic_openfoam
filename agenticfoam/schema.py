"""The data model that flows through the harness.

    CaseSpec     -- the intent (what the planner distils from the NL request)
    CaseState    -- mutable on-disk status of one case
    ToolCall     -- one action the executor took
    StageResult  -- outcome of one attempt at one stage
    RunLedger    -- the whole run: the CaseSpec + every StageResult + the verdict

The RunLedger is the audit trail: every LLM decision is recorded with the tool
calls it made and the gate that judged it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Lid-driven cavity reference scales: lid speed U = 1 m/s, cavity side L = 0.1 m.
LID_SPEED = 1.0
CAVITY_SIZE = 0.1


def nu_for_reynolds(re: float) -> float:
    """nu = U * L / Re."""
    return LID_SPEED * CAVITY_SIZE / re


class Stage(str, Enum):
    PRE = "pre"          # generate + check the mesh
    SETUP = "setup"      # viscosity, BCs, time controls
    SOLVE = "solve"      # run the solver to a converged / bounded state
    POST = "post"        # sample lines, VTK, extract quantities of interest


class MeshSpec(BaseModel):
    counts: tuple[int, int, int] = (20, 20, 1)
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0)


class CaseSpec(BaseModel):
    name: str
    reynolds: float
    nu: float
    mesh: MeshSpec = Field(default_factory=MeshSpec)
    solver: str = "icoFoam"
    end_time: float = 0.5
    delta_t: float = 0.005
    write_interval_steps: int = 20
    quantities: list[str] = Field(default_factory=lambda: ["primary_vortex"])
    assumptions: list[str] = Field(default_factory=list)

    @classmethod
    def from_reynolds(cls, name: str, re: float, **kw: Any) -> "CaseSpec":
        return cls(name=name, reynolds=re, nu=nu_for_reynolds(re), **kw)


class CaseState(BaseModel):
    spec: CaseSpec
    path: str
    stage: Stage = Stage.PRE
    status: str = "pending"              # pending | running | passed | failed | escalated
    time_dirs: list[str] = Field(default_factory=list)
    last_checkpoint: str | None = None   # path to a snapshot dir
    stage_attempts: int = 0              # recovery attempts spent in the current stage


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any]               # trimmed by the caller before storing


class StageResult(BaseModel):
    case: str
    stage: Stage
    attempt: int
    status: str                         # passed | failed
    gate: str                          # human-readable gate verdict
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    rationale: str = ""                 # the executor's stated reasoning
    elapsed_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class RunLedger(BaseModel):
    request: str
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = ""
    specs: dict[str, CaseSpec] = Field(default_factory=dict)
    stage_results: list[StageResult] = Field(default_factory=list)
    recoveries: list[str] = Field(default_factory=list)          # one line per recovery
    verdict: str | None = None          # pass | fail | escalated | error
    summary: str = ""

    def add_case(self, spec: CaseSpec) -> None:
        self.specs[spec.name] = spec

    def record(self, r: StageResult) -> None:
        self.stage_results.append(r)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, default=str))
        return path
