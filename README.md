# agenticFOAM

A stage-gated **agentic harness** that drives an OpenFOAM tutorial — the
lid-driven cavity (`icoFoam`, User Guide §2.1) — from a one-line request to a
converged case, with no human in the loop.

> *"Solve the lid-driven cavity at Re = 100 and tell me where the primary vortex
> sits."*

→ the harness plans the case, builds the mesh, sets the physics, runs the solver
**to a convergence criterion it judges itself**, and extracts the vortex —
recovering from mesh-quality issues and numerical instability along the way.

**Design rationale and diagrams: [`DESIGN.md`](DESIGN.md).**

---

## Quick start

```bash
pip install -e .                       # httpx, pydantic, pyyaml, rich
echo "OPENROUTER_API_KEY=sk-or-..." > .env
python -m agenticfoam.demo             # the reference Re=100 run
```

Any request works:

```bash
python -m agenticfoam.demo "solve the cavity at Re=400 on a 30x30 mesh"
python -m agenticfoam.demo -m anthropic/claude-sonnet-4.5   # pick the model
```

Requires OpenFOAM v2412 on the machine (the harness shells out to `blockMesh`,
`icoFoam`, `postProcess`). Point `AGENTICFOAM_BASHRC` at a different install if
needed.

### Model configuration

| where | how |
|---|---|
| `.env` | `AGENTICFOAM_MODEL=openai/gpt-4.1` |
| env var | `AGENTICFOAM_MODEL=... python -m agenticfoam.demo` |
| CLI | `python -m agenticfoam.demo -m <slug>` |

Any OpenRouter slug. Endpoint is `AGENTICFOAM_BASE_URL` (default OpenRouter); key
from `OPENROUTER_API_KEY` or `OPENAI_API_KEY`. The deterministic spine carries
the run, so weaker models still work — they just take more recovery iterations.

---

## What you see

A persistent stage checklist pinned to the bottom of the terminal, with the
agent's reasoning and actions scrolling above it:

```
▸ SOLVE  run to convergence   retry 2/5
   » Courant hit 1.85, above the limit; cutting deltaT to hold Co ≈ 0.5.
   ✓ set_config       controlDict · deltaT  0.005 → 0.00135
   ✓ run_solver       t=1.1  ·  Co 0.50  ·  U resid 8.2e-06  ·  ran clean
   ✓ gate PASS  converged: U residual < 1e-5, solver no longer iterating U
   ↻ SOLVE on cavity_re100: converged after 2 attempts
──────────────────────────────────────────────────────────────
╭─ agenticFOAM · cavity_re100 ───────────────────────────────╮
│ ✓  pre      mesh OK · 400 cells · non-orthogonality 0°     │
│ ✓  setup    fields present and consistent with the mesh    │
│ ↻  solve    retry 2/5 · Courant 1.85 (limit 1.0)            │
│ ·  post                                                    │
│ recoveries 1   ·   tool calls 21   ·   1m40s                │
╰────────────────────────────────────────────────────────────╯
```

Every run also writes `runs/latest/ledger.json` — the full audit trail: the plan
and its assumptions, and for every stage attempt the tool-call trace, the
executor's rationale, the gate verdict and the metrics it judged on.

---

## Repo layout

```
agenticfoam/
  schema.py        CaseSpec · CaseState · StageResult · RunLedger  (the data model)
  foam.py          run an OpenFOAM binary · parse solver logs · dict edits · snapshot/rollback
  tools.py         the action space + schema derivation + dispatcher
  analysis.py      streamfunction vortex locator
  gates.py         the 4 stage gates + the regime-aware convergence classifier
  stages.py        the per-stage executor sub-loop (brief → tool calls → gate → retry)
  agents.py        planner (request → CaseSpec)
  orchestrator.py  top-level flow · recovery ladder · the ledger
  llm.py           provider-neutral OpenRouter client (httpx, no SDK)
  report.py        the pinned-status-bar terminal reporter (+ plain fallback)
  demo.py          the showcase entrypoint
cases/cavity/      pristine vendored tutorial case (0/ constant/ system/)
runs/              ledgers and case snapshots (gitignored)
```

---

## Tests / verification

The deterministic pieces were checked against real OpenFOAM output:

- convergence classifier — correct verdict on Re = 10 / 100-short / 100-long /
  10⁴-turbulent / `Δt`-too-large;
- vortex locator — matches Ghia, Ghia & Shin (1982) to ~1 % at Re = 100.
