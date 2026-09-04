"""Post-processing analysis:

  * primary_vortex -- vortex centre = interior extremum of the stream function
                      psi(x,y) = integral_0^y u_x dy' (trapezoid; graded-mesh safe)
"""
from __future__ import annotations

import re
from pathlib import Path

from .schema import CAVITY_SIZE
from .foam import latest_time

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def _read_internal(path: Path):
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\n\(", txt)
    if not m:
        mu = re.search(r"internalField\s+uniform\s+(.+?);", txt)
        raise RuntimeError(
            f"{path}: internalField is uniform ({mu.group(1) if mu else '?'}), not a solved field"
        )
    kind, n = m.group(1), int(m.group(2))
    depth, i = 1, m.end()
    while depth:
        depth += (txt[i] == "(") - (txt[i] == ")")
        i += 1
    body = txt[m.end() : i - 1]
    if kind == "scalar":
        vals = [float(x) for x in re.findall(_NUM, body)]
        assert len(vals) == n, (path, len(vals), n)
        return vals
    tup = re.findall(r"\(\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\)", body)
    assert len(tup) == n, (path, len(tup), n)
    return [(float(a), float(b), float(c)) for a, b, c in tup]


class Grid:
    """Structured view of a solved NX x NY x 1 cavity field."""

    def __init__(self, case: Path, time: str):
        C = _read_internal(case / time / "C")
        U = _read_internal(case / time / "U")
        self.xs = sorted({round(c[0], 7) for c in C})
        self.ys = sorted({round(c[1], 7) for c in C})
        self.nx, self.ny = len(self.xs), len(self.ys)
        ix = {v: i for i, v in enumerate(self.xs)}
        iy = {v: j for j, v in enumerate(self.ys)}
        self.ux = [[0.0] * self.nx for _ in range(self.ny)]
        for c, u in zip(C, U):
            self.ux[iy[round(c[1], 7)]][ix[round(c[0], 7)]] = u[0]
        self.time = time

    def streamfunction(self) -> list[list[float]]:
        psi = [[0.0] * self.nx for _ in range(self.ny)]
        for i in range(self.nx):
            for j in range(1, self.ny):
                psi[j][i] = psi[j - 1][i] + 0.5 * (self.ux[j][i] + self.ux[j - 1][i]) * (
                    self.ys[j] - self.ys[j - 1]
                )
        return psi


def _grid(case: str | Path, time: str | None) -> Grid:
    case = Path(case)
    t = time or latest_time(case)
    if t is None or t == "0":
        raise RuntimeError("no solved time directory")
    if not (case / t / "C").is_file():
        raise RuntimeError(f"{t}/C missing -- run postProcess writeCellCentres first")
    return Grid(case, t)


def primary_vortex(case: str | Path, time: str | None = None) -> dict:
    g = _grid(case, time)
    psi = g.streamfunction()
    best = (0, 0, 0.0)
    for j in range(1, g.ny - 1):
        for i in range(1, g.nx - 1):
            if abs(psi[j][i]) > abs(best[2]):
                best = (i, j, psi[j][i])
    return {
        "time": g.time,
        "x": round(g.xs[best[0]], 6),
        "y": round(g.ys[best[1]], 6),
        "x_norm": round(g.xs[best[0]] / CAVITY_SIZE, 4),
        "y_norm": round(g.ys[best[1]] / CAVITY_SIZE, 4),
        "psi_extremum": round(best[2], 6),
        "nx": g.nx,
        "ny": g.ny,
    }
