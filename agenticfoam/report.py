"""Human-facing progress reporting.

The harness emits typed events (stage_start, rationale, tool, tool_result, gate,
recovery, ...) and a Reporter renders them. Two renderers:

  PlainReporter -- one line per event; for logs, pipes, CI, background runs.
  RichReporter  -- a persistent stage checklist pinned to the bottom of the
                   terminal, with a readable narrative scrolling above it.

`make_reporter()` picks Rich for an interactive TTY, Plain otherwise.
"""
from __future__ import annotations

import re
import textwrap
import time
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .schema import CaseSpec, RunLedger, Stage

_STAGES = [Stage.PRE, Stage.SETUP, Stage.SOLVE, Stage.POST]
_ICON = {"pending": "·", "running": "▶", "passed": "✓", "failed": "✗", "recover": "↻"}
_STYLE = {"pending": "dim", "running": "yellow", "passed": "green", "failed": "red", "recover": "yellow"}
_DESC = {"pre": "mesh", "setup": "fields & time controls", "solve": "run to convergence",
         "post": "extract quantities"}

# tools whose humanised result already contains everything the args would say
_ARGS_HIDDEN = {"set_mesh", "set_config", "run_blockmesh", "run_solver", "run_postprocess",
                "mesh_quality", "extract_primary_vortex", "clone_case"}


def _num(x, places=0):
    try:
        f = float(x)
        return f"{int(round(f))}" if places == 0 and f == int(f) else f"{f:.{places}g}"
    except (TypeError, ValueError):
        return str(x)


# --------------------------------------------------------------------------- #
# Turn a raw tool result into a short human sentence
# --------------------------------------------------------------------------- #
def humanize(name: str, args: dict, res: Any) -> str:
    if isinstance(res, dict) and res.get("error"):
        return f"[red]{str(res['error']).splitlines()[0][:120]}[/red]"
    r = res if isinstance(res, dict) else {"value": res}

    if name == "run_solver":
        bits = [f"t={r.get('last_time')}", f"Co {_num(r.get('max_courant'), 2)}"]
        if r.get("u_initial_residual") is not None:
            bits.append(f"U resid {r['u_initial_residual']:.1e}")
        bits.append("[red]diverged[/red]" if r.get("diverged") else "ran clean")
        return "  ·  ".join(bits)
    if name == "run_blockmesh":
        return f"{_num(r.get('n_cells'))} cells" + ("" if r.get("ok") else "  ·  [red]FAILED[/red]")
    if name == "mesh_quality":
        sk = r.get("max_skewness") or 0
        return (f"non-orthogonality {_num(r.get('max_non_orthogonality') or 0)}°  ·  "
                f"aspect {_num(r.get('max_aspect_ratio') or 0, 2)}  ·  "
                f"skew {0 if sk < 1e-6 else _num(sk, 1)}")
    if name == "set_config":
        return f"{r.get('file')} · {r.get('key')}  {r.get('old')} → [bold]{r.get('new')}[/bold]"
    if name == "set_mesh":
        c = r.get("new", {}).get("counts", ["?", "?"])
        g = r.get("new", {}).get("grading", [1, 1])
        graded = any(abs(float(x) - 1) > 1e-9 for x in g[:2])
        return f"{c[0]}×{c[1]}" + (f", grading {_num(g[0],2)}·{_num(g[1],2)}" if graded else ", uniform")
    if name == "clone_case":
        return f"case {r.get('dest')} ready"
    if name == "run_postprocess":
        return "wrote cell centres" if r.get("produced") else "done"
    if name == "extract_primary_vortex":
        return f"vortex at x/L {r.get('x_norm', '?')}, y/L {r.get('y_norm', '?')}"
    if name == "read_config":
        return f"{args.get('key')} = [bold]{r.get('value')}[/bold]"
    if name == "list_times":
        return f"times {', '.join(r.get('times', []))}"
    if name == "read_log":
        return f"{args.get('tail_lines', 40)} lines of {args.get('name')}"
    keys = [k for k in ("ok", "value", "new") if k in r]
    return ", ".join(f"{k}={r[k]}" for k in keys) or "done"


def fmt_args(name: str, a: dict) -> str:
    if name in _ARGS_HIDDEN:
        return ""
    shown = {k: v for k, v in a.items() if k not in ("case", "target", "source")}
    return "  ".join(f"[dim]{k}[/dim] {v}" for k, v in shown.items())


# some open models leak scratchpad / channel control tokens into content
_NOISE = re.compile(r"<\|?/?(channel|tool_call|thought|end|im_start|im_end)[^>]*\|?>|`{1,3}", re.I)


def _clean_rationale(text: str) -> str:
    t = _NOISE.sub("", text or "").strip(" \n\t{}[]`")
    lines = [ln.strip() for ln in t.splitlines() if len(ln.strip()) > 3
             and ln.strip().lower() not in ("thought", "thinking", "reasoning")]
    t = " ".join(lines)
    if len(t) < 4:
        return ""
    # first sentence only -- the actionable bit -- capped to one printed line
    first = t.split(". ")[0].rstrip(".")
    first = first[:118].rsplit(" ", 1)[0] + "…" if len(first) > 120 else first
    return first + "."


# --------------------------------------------------------------------------- #
class Reporter:
    """Plain, line-per-event reporter (also the base class)."""

    _IND = "   "   # base indent for everything inside a stage

    def __init__(self, console: Console | None = None):
        self.con = console or Console()
        self.t0 = time.time()
        self.state: dict[str, str] = {s.value: "pending" for s in _STAGES}
        self.case = ""
        self.n_tools = 0
        self.n_recoveries = 0

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # -- events ----------------------------------------------------------
    def run_start(self, request: str, model: str):
        self.con.rule("[bold]agenticFOAM[/bold]", style="cyan")
        self.con.print(f"[dim]request[/dim]  {request}")
        self.con.print(f"[dim]model  [/dim]  {model}")

    def plan(self, spec: CaseSpec):
        self.con.print(f"\n[bold cyan]▸ PLAN[/bold cyan]  "
                       f"Re {spec.reynolds:g} · {spec.mesh.counts[0]}×{spec.mesh.counts[1]} · "
                       f"{spec.solver} · ν={spec.nu:g}")
        for a in spec.assumptions:
            self.con.print(f"{self._IND}[dim]· {a}[/dim]")

    def case_start(self, spec: CaseSpec):
        self.case = spec.name
        self.state = {s.value: "pending" for s in _STAGES}
        self.con.print()
        self.con.rule(f"[bold]{spec.name}[/bold]  ·  Re {spec.reynolds:g} · "
                      f"{spec.mesh.counts[0]}×{spec.mesh.counts[1]} · {spec.solver}", style="cyan")

    def stage_start(self, stage: Stage, attempt: int, max_attempts: int):
        self.state[stage.value] = "running" if attempt == 1 else "recover"
        tag = "" if attempt == 1 else f"   [yellow]retry {attempt}/{max_attempts}[/yellow]"
        self.con.print(f"\n[bold cyan]▸ {stage.value.upper()}[/bold cyan]  "
                       f"[dim]{_DESC.get(stage.value, '')}[/dim]{tag}")

    def note(self, text: str):
        self.con.print(f"{self._IND}[dim]{text}[/dim]")

    def crash(self, stage: Stage, err: str, tb: str):
        self.state[stage.value] = "failed"
        self.con.print(f"{self._IND}[red]✗ {stage.value} crashed:[/red] {err}")
        self.con.print(f"[dim]{tb[-600:]}[/dim]")

    def rationale(self, text: str):
        clean = _clean_rationale(text)
        if not clean:
            return
        wrapped = textwrap.fill(
            clean, width=max(40, self.con.width - 2),
            initial_indent=f"{self._IND}» ", subsequent_indent=f"{self._IND}  ",
        )
        self.con.print(f"[dim italic]{wrapped}[/dim italic]")

    def tool(self, name: str, args: dict):
        self.n_tools += 1          # printed on completion (see tool_result)

    def tool_result(self, name: str, args: dict, ok: bool, res: Any):
        glyph = "[green]✓[/green]" if ok else "[red]✗[/red]"
        extra = fmt_args(name, args)
        extra = f"  {extra}" if extra else ""
        self.con.print(f"{self._IND}{glyph} [cyan]{name:<16}[/cyan] {humanize(name, args, res)}{extra}")

    def gate(self, stage: Stage, passed: bool, reason: str):
        self.state[stage.value] = "passed" if passed else "failed"
        if passed:
            self.con.print(f"{self._IND}[green bold]✓ gate PASS[/green bold]  [dim]{reason}[/dim]")
        else:
            self.con.print(f"{self._IND}[yellow bold]✗ gate FAIL[/yellow bold]  [dim]{reason}[/dim]")

    def recovery(self, text: str):
        self.n_recoveries += 1
        self.con.print(f"{self._IND}[yellow]↻ {text}[/yellow]")

    def rollback(self, case: str, stage: Stage):
        self.con.print(f"{self._IND}[dim]↩ rolled {case} back to {stage.value} entry[/dim]")

    def run_end(self, ledger: RunLedger):
        colour = {"pass": "green", "escalated": "yellow", "error": "red"}.get(ledger.verdict, "white")
        self.con.print()
        self.con.rule(f"[{colour} bold]{(ledger.verdict or '?').upper()}[/{colour} bold]", style=colour)
        self.con.print(ledger.summary + "\n")
        if ledger.recoveries:
            self.con.print(f"[dim]recoveries[/dim] {len(ledger.recoveries)}")
            for rcv in ledger.recoveries:
                self.con.print(f"           [yellow]↻[/yellow] {rcv}")
        self.con.print(f"[dim]tool calls[/dim] {sum(len(r.tool_calls) for r in ledger.stage_results)}"
                       f"  ·  [dim]stages[/dim] {len(ledger.stage_results)}")


# --------------------------------------------------------------------------- #
class RichReporter(Reporter):
    """Persistent stage checklist pinned to the bottom of the terminal; the
    narrative scrolls in the space above it (Rich's Live redirects console
    output above the live region). Fixed height + manual refresh so the bar
    never 'walks' or flickers."""

    _BAR_HEIGHT = 8        # 4 stage rows + spacer + footer + 2 border -> constant
    _HINT_W = 44

    def __enter__(self):
        self._hints: dict[str, str] = {}
        # Push the bar to the literal bottom row from the first frame (like a
        # shell prompt / Claude Code's input): fill the screen with blank lines,
        # then Live pins the bar there and scrolls everything else above it.
        pad = max(0, self.con.size.height - self._BAR_HEIGHT - 1)
        self.con.print("\n" * pad, end="")
        # auto_refresh off: we redraw exactly once per event, after the scrolling
        # line for that event has printed -- no background thread racing prints.
        self._live = Live(self._bar(), console=self.con, auto_refresh=False,
                          transient=False, vertical_overflow="visible")
        self._live.__enter__()
        self._live.refresh()
        return self

    def __exit__(self, *exc):
        self._live.update(self._bar(), refresh=True)
        self._live.__exit__(*exc)
        return False

    def _bar(self) -> Panel:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=10)
        grid.add_column(width=self._HINT_W)
        for s in _STAGES:
            st = self.state[s.value]
            hint = self._hints.get(s.value, "")
            hint = hint if len(hint) <= self._HINT_W else hint[: self._HINT_W - 1] + "…"
            grid.add_row(Text(f"{_ICON[st]}  {s.value}", style=_STYLE[st]),
                         Text(hint, style="dim"))
        elapsed = int(time.time() - self.t0)
        foot = Text(f"recoveries {self.n_recoveries}   ·   tool calls {self.n_tools}"
                    f"   ·   {elapsed // 60}m{elapsed % 60:02d}s", style="dim")
        title = f"agenticFOAM · {self.case}" if self.case else "agenticFOAM"
        return Panel(Group(grid, Text(""), foot), title=title, title_align="left",
                     border_style="cyan", height=self._BAR_HEIGHT)

    def _refresh(self):
        self._live.update(self._bar(), refresh=True)

    def case_start(self, spec):
        self._hints = {}
        super().case_start(spec)
        self._refresh()

    def stage_start(self, stage, attempt, max_attempts):
        super().stage_start(stage, attempt, max_attempts)
        if attempt > 1:
            self._hints[stage.value] = f"retry {attempt}/{max_attempts}…"
        self._refresh()

    def tool(self, name, args):
        super().tool(name, args)
        rs = _running_stage(self.state)
        if rs:
            self._hints[rs] = f"{name}…"
        self._refresh()

    def tool_result(self, name, args, ok, res):
        super().tool_result(name, args, ok, res)
        rs = _running_stage(self.state)
        if rs:
            self._hints[rs] = _plain(humanize(name, args, res))
        self._refresh()

    def gate(self, stage, passed, reason):
        super().gate(stage, passed, reason)
        self._hints[stage.value] = _plain(reason)
        self._refresh()

    def recovery(self, text):
        super().recovery(text)
        self._refresh()

    def crash(self, stage, err, tb):
        super().crash(stage, err, tb)
        self._refresh()

    def run_end(self, ledger):
        self._refresh()
        super().run_end(ledger)


def _running_stage(state: dict[str, str]) -> str | None:
    for k, v in state.items():
        if v in ("running", "recover"):
            return k
    return None


def _plain(markup: str) -> str:
    """Strip Rich markup for use in the fixed-width status bar."""
    return Text.from_markup(markup).plain


def make_reporter(console: Console | None = None) -> Reporter:
    con = console or Console()
    return RichReporter(con) if con.is_terminal else Reporter(con)
