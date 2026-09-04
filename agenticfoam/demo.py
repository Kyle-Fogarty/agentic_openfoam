"""The showcase run.

    python -m agenticfoam.demo
    python -m agenticfoam.demo "solve the lid-driven cavity at Re=400 on a 30x30 mesh"

Default request is deliberately underspecified. Expected trajectory:

  plan   Re=100, 20x20 (assumed), icoFoam, exploratory endTime 0.5
  PRE    mesh OK
  SETUP  cold start, time controls set
  SOLVE  classifier -> NOT_YET -> executor extends endTime -> CONVERGED   (recovery)
  POST   primary vortex extracted -> PASS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .llm import DEFAULT_MODEL
from .orchestrator import run
from .report import make_reporter

DEFAULT_REQUEST = "Solve the lid-driven cavity at Re = 100 and tell me where the primary vortex sits."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agenticfoam.demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", nargs="?", default=DEFAULT_REQUEST,
                    help="natural-language simulation request")
    ap.add_argument("-m", "--model", default=None,
                    help=f"OpenRouter model slug (default: $AGENTICFOAM_MODEL or {DEFAULT_MODEL})")
    ap.add_argument("--root", default="runs/latest", help="output directory")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    out_dir = Path(args.root)
    con = Console()
    reporter = make_reporter(con)

    with reporter:
        ledger = run(args.request, model=args.model, root=out_dir, report=reporter)
    path = ledger.write(out_dir / "ledger.json")
    con.print(f"[dim]ledger[/dim]     {path}")
    return 0 if ledger.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
