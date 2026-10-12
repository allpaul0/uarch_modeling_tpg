"""
analysis/static_args.py — which calls of the TPG programs have arguments
known at compile time, and which of those take an early-exit guard.

Why
---
The cost of ``fixedpt_exp_wdiv32``, ``fixedpt_ln_wdiv32`` … depends on the
argument: ``exp`` returns at once for ``fp == 0`` or out of range, ``ln`` for
``x <= 0``, the protected divide for ``|B| < 256``.  A ``jal`` count cannot
see that, so one weight has to average cheap and expensive calls.

Part of the argument is visible in the C code, though.  A program starts with
``reg[0..7] = 0``, so a register it reads before writing is the constant 0;
``X - X`` is 0 whatever X; ``fixedpt_mul(X, 0)`` is 0; ``exp(0)`` is
FIXEDPT_ONE, and so on.  This module propagates those constants through each
program and classifies every call:

  * ``static_guard`` — every argument is a known constant and it trips the
    function's guard: the call takes the early exit (a cheap, fixed cost).
  * ``static_full``  — every argument is a known constant, no guard: the
    path is still fixed, only longer.
  * ``same_operand`` — the arguments are the same unknown value
    (``protected_fixedpt_div_32(X, X)``): the result is known up to the guard,
    the cost is not.
  * ``dynamic``      — at least one argument depends on the inputs.

A call is also flagged ``dead`` when its result never reaches ``reg[0]``, the
value the program returns.  With ``noipa`` the call is still executed (the
compiler cannot prove it has no side effect); with inlined functions the
compiler removes it.

Constants are folded with the real fixedptc.h, compiled for the host, so the
values are the ones the target computes.  The one exception is the protected
32-bit divide, emulated here because ``INT_MIN / -1`` traps on x86 but
returns INT_MIN on RISC-V.

Usage
-----
    from analysis.static_args import static_args
    static_args("db_fixedpt_iset32_noipa", uarch="cv32e40x_im1_zba_zbb").print()

or ``python main.py static-args --db <path> [--uarch <name>] [--header <fixedptc.h>]``.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from analysis.inventory import _PROG_RE, _TEAM_RE, _USE_RE

if TYPE_CHECKING:
    from classes.database import Database


# ---------------------------------------------------------------------------
# Fixed-point semantics (Q16.16, int32, as on the target)
# ---------------------------------------------------------------------------

FBITS = 16
ONE = 1 << FBITS
INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1


def _wrap32(v: int) -> int:
    return ((v + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def _rconst(r: float) -> int:
    """fixedpt_rconst(R): round half away from zero, then C truncation."""
    return int(r * ONE + (0.5 if r >= 0 else -0.5))


def _abs32(v: int) -> int:
    """fixedpt_abs, including abs(INT_MIN) == INT_MIN."""
    return _wrap32(-v) if v < 0 else v


def _mul(a: int, b: int) -> int:
    return _wrap32((a * b) >> FBITS)


def _div32(a: int, b: int) -> int:
    """protected_fixedpt_div_32 with RISC-V division semantics."""
    if _abs32(b) < (1 << (FBITS // 2)):
        return 0
    num = _wrap32(a << (FBITS // 2))
    den = b >> (FBITS // 2)
    if den == -1 and num == INT_MIN:
        return INT_MIN
    q = abs(num) // abs(den)
    return _wrap32(q if (num < 0) == (den < 0) else -q)


def _log2(n: int) -> int:
    return 0 if n <= 0 else n.bit_length() - 1


def _pow2(b: int) -> int:
    return 0 if b < 0 or b > 30 else 1 << b


_SHIM = r"""
#include <stdint.h>
#include "fixedptc.h"
fixedpt sa_exp(fixedpt a) { return fixedpt_exp_wdiv32(a); }
fixedpt sa_ln(fixedpt a)  { return fixedpt_ln_wdiv32(a); }
fixedpt sa_sin(fixedpt a) { return fixedpt_sin(a); }
fixedpt sa_cos(fixedpt a) { return fixedpt_cos(a); }
"""


class _Folder:
    """Evaluates the fixedpt library on constants, with the real header."""

    def __init__(self, header: Path):
        self.header = header
        tmp = Path(tempfile.mkdtemp(prefix="static_args_"))
        (tmp / "shim.c").write_text(_SHIM)
        (tmp / "fixedptc.h").write_text(header.read_text())
        subprocess.run(["gcc", "-O1", "-w", "-shared", "-fPIC", "-o",
                        str(tmp / "shim.so"), str(tmp / "shim.c")],
                       check=True, cwd=tmp)
        lib = ctypes.CDLL(str(tmp / "shim.so"))
        self._c = {}
        for name in ("exp", "ln", "sin", "cos"):
            fn = getattr(lib, f"sa_{name}")
            fn.argtypes, fn.restype = [ctypes.c_int32], ctypes.c_int32
            self._c[name] = fn

    def call(self, func: str, args: list[int]) -> int:
        c = self._c
        if func == "fixedpt_exp_wdiv32":       return c["exp"](args[0])
        if func == "fixedpt_ln_wdiv32":        return c["ln"](args[0])
        if func == "fixedpt_sin":              return c["sin"](args[0])
        if func == "fixedpt_cos":              return c["cos"](args[0])
        if func == "fixedpt_tan_wdiv32":       return _div32(c["sin"](args[0]), c["cos"](args[0]))
        if func == "protected_fixedpt_div_32": return _div32(args[0], args[1])
        if func == "fixedpt_mul":              return _mul(args[0], args[1])
        if func == "f_log2":                   return _log2(args[0])
        if func == "f_pow2":                   return _pow2(args[0])
        raise KeyError(func)

    def guard(self, func: str, args: list[int]) -> bool:
        """True when constant ``args`` make ``func`` take its early exit."""
        if func == "fixedpt_exp_wdiv32":
            x = args[0]
            return x == 0 or x > _rconst(10.39) or x < _rconst(-11.09)
        if func == "fixedpt_ln_wdiv32":
            return args[0] <= 0
        if func == "protected_fixedpt_div_32":
            return _abs32(args[1]) < (1 << (FBITS // 2))
        if func == "fixedpt_tan_wdiv32":       # guard of the inner divide by cos
            return _abs32(self._c["cos"](args[0])) < (1 << (FBITS // 2))
        if func == "f_log2":
            return args[0] <= 0
        if func == "f_pow2":
            return args[0] < 0 or args[0] > 30
        return False                           # sin, cos, mul: no guard


# ---------------------------------------------------------------------------
# Constant propagation over one program
# ---------------------------------------------------------------------------

# Functions that stay a jal in the noipa build.  The others (fixedpt_mul,
# f_log2, f_pow2) are always inlined, so the compiler folds their constant
# cases itself in every build; they are reported but not counted as calls.
CALLED = frozenset({
    "fixedpt_exp_wdiv32", "fixedpt_ln_wdiv32", "protected_fixedpt_div_32",
    "fixedpt_sin", "fixedpt_cos", "fixedpt_tan_wdiv32",
})

STATIC_GUARD = "static_guard"
STATIC_FULL  = "static_full"
SAME_OPERAND = "same_operand"
DYNAMIC      = "dynamic"
CLASSES = (STATIC_GUARD, STATIC_FULL, SAME_OPERAND, DYNAMIC)

_OPDECL_RE = re.compile(r"^fixedpt\s+(op\d+)\s*=\s*(in\d+\[\d+\]|reg\[(\d+)\])\s*;$")
_ASSIGN_RE = re.compile(r"^reg\[(\d+)\]\s*=\s*(.+);$")
_BIN_RE    = re.compile(r"^(op\d+)\s*([+-])\s*(op\d+)$")
_TERN_RE   = re.compile(r"^\((op\d+)\s*>\s*(op\d+)\)\s*\?\s*(op\d+)\s*:\s*(op\d+)$")
_FUNC_RE   = re.compile(r"^(\w+)\((op\d+)(?:\s*,\s*(op\d+))?\)$")


@dataclass
class CallSite:
    """One function call of one program."""
    program: str
    func: str
    cls: str
    dead: bool
    args: tuple          # int for a known constant, None for an input-dependent value


class _Values:
    """
    Abstract values: ("c", int) is a known constant, ("s", id) an unknown
    value.  Unknown values are hash-consed, so two expressions that compute
    the same thing from the same inputs get the same id — that is what makes
    ``X - X`` and ``div(X, X)`` recognisable.
    """

    def __init__(self):
        self._ids: dict[tuple, int] = {}

    def sym(self, *key) -> tuple:
        return ("s", self._ids.setdefault(key, len(self._ids)))


def analyse_program(name: str, body: str, folder: _Folder) -> tuple[list[CallSite], int]:
    """
    Call sites of one program, and the number of statements not understood.
    """
    V = _Values()
    regs = [("c", 0)] * 8
    ops: dict[str, tuple] = {}
    op_reg: dict[str, int | None] = {}
    stmts: list[dict] = []           # for the liveness pass
    unparsed = 0

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line in ("{", "}") or line.startswith(("return", "fixedpt reg[")):
            continue
        m = _OPDECL_RE.match(line)
        if m:
            op, src, r = m.groups()
            ops[op] = regs[int(r)] if r is not None else V.sym("in", src)
            op_reg[op] = int(r) if r is not None else None
            continue
        m = _ASSIGN_RE.match(line)
        if not m:
            unparsed += 1
            continue
        dst, expr = int(m.group(1)), m.group(2).strip()
        used: list[str] = []
        call = None

        if (b := _BIN_RE.match(expr)):
            a, sign, c = b.groups(); used = [a, c]
            x, y = ops[a], ops[c]
            if x[0] == y[0] == "c":
                val = ("c", _wrap32(x[1] + y[1] if sign == "+" else x[1] - y[1]))
            elif sign == "-" and x == y:
                val = ("c", 0)
            elif y == ("c", 0):
                val = x
            elif sign == "+" and x == ("c", 0):
                val = y
            else:
                val = V.sym(sign, *(sorted((x, y)) if sign == "+" else (x, y)))
        elif (t := _TERN_RE.match(expr)):
            a, c, u, w = t.groups(); used = [a, c, u, w]
            x, y = ops[a], ops[c]
            if x[0] == y[0] == "c":
                val = ops[u] if x[1] > y[1] else ops[w]
            elif x == y:                           # X > X is false
                val = ops[w]
            elif ops[u] == ops[w]:
                val = ops[u]
            else:
                val = V.sym("?:", x, y, ops[u], ops[w])
        elif (f := _FUNC_RE.match(expr)):
            func = f.group(1); used = [o for o in f.groups()[1:] if o]
            args = [ops[o] for o in used]
            if all(v[0] == "c" for v in args):
                consts = [v[1] for v in args]
                val = ("c", folder.call(func, consts))
                cls = STATIC_GUARD if folder.guard(func, consts) else STATIC_FULL
            elif func == "fixedpt_mul" and ("c", 0) in args:
                val, cls = ("c", 0), STATIC_FULL
            else:
                val = V.sym(func, *args)
                cls = SAME_OPERAND if len(args) > 1 and args[0] == args[1] else DYNAMIC
            call = CallSite(name, func, cls, False,
                            tuple(v[1] if v[0] == "c" else None for v in args))
        else:
            unparsed += 1
            continue

        regs[dst] = val
        stmts.append(dict(dst=dst, srcs={op_reg[o] for o in used} - {None}, call=call))

    # Liveness: what reaches "return reg[0]".
    live = {0}
    for s in reversed(stmts):
        if s["dst"] in live:
            live.discard(s["dst"])
            live |= s["srcs"]
        elif s["call"] is not None:
            s["call"].dead = True
    return [s["call"] for s in stmts if s["call"] is not None], unparsed


# ---------------------------------------------------------------------------
# Database-wide report
# ---------------------------------------------------------------------------

def _find_header(source_path: str) -> Path | None:
    """pipeline/parameters/fixedptc.h of the tpg_nextflow tree holding the TPG."""
    for parent in Path(source_path).parents:
        cand = parent / "pipeline" / "parameters" / "fixedptc.h"
        if cand.exists():
            return cand
    return None


def _codegen(source_path: str) -> tuple[Path, Path] | None:
    root = Path(source_path) / "outLogs"
    for sub in ("codegen_TeamsInstrumented", "codegen"):
        prog_f, tpg_f = root / sub / "TPG_program.h", root / sub / "TPG.c"
        if prog_f.exists() and tpg_f.exists():
            return prog_f, tpg_f
    return None


@dataclass
class StaticArgsReport:
    db_name: str
    header: Path
    n_tpgs: int = 0
    no_codegen: int = 0
    unparsed: int = 0
    # one entry per call site after expanding programs into their teams
    sites: list[dict] = field(default_factory=list)
    # (tpg, team) -> measured latency on the requested uarch
    latency: dict[tuple[str, int], float] = field(default_factory=dict)
    uarch: str | None = None

    def print(self) -> None:
        bar = "─" * 96
        print(f"\n{bar}\n  STATIC ARGUMENTS  {self.db_name}\n{bar}")
        print(f"  header: {self.header}")
        called = [s for s in self.sites if s["func"] in CALLED]
        print(f"  {self.n_tpgs} TPGs ({self.no_codegen} without codegen), "
              f"{len({(s['tpg'], s['team']) for s in called})} teams with calls, "
              f"{self.unparsed} statements not understood")

        print("\n  CALL SITES (programs expanded into their teams)")
        print(f"  {'function':28}{'sites':>7}" + "".join(f"{c:>14}" for c in CLASSES)
              + f"{'dead':>8}")
        by_func = defaultdict(list)
        for s in self.sites:
            by_func[s["func"]].append(s)
        rows = sorted(by_func.items(), key=lambda kv: (kv[0] not in CALLED, -len(kv[1])))
        for func, ss in rows[:sum(f in CALLED for f in by_func)] + [("ALL CALLS", called)] \
                + rows[sum(f in CALLED for f in by_func):]:
            if not ss:               # a database whose programs call nothing
                continue
            n = len(ss); cnt = Counter(s["cls"] for s in ss)
            tag = "" if func in CALLED or func == "ALL CALLS" else " (inlined)"
            print(f"  {func + tag:28}{n:>7}" + "".join(f"{cnt[c] / n:>14.1%}" for c in CLASSES)
                  + f"{sum(s['dead'] for s in ss) / n:>8.1%}")
        print("  static_guard : constant arguments that take the early exit")
        print("  static_full  : constant arguments, full path (cost fixed but not small)")
        print("  same_operand : f(X, X) with X unknown        dead : result never reaches reg[0]")

        teams = defaultdict(set)
        tpgs = defaultdict(set)
        for s in called:
            teams[s["cls"]].add((s["tpg"], s["team"]))
            tpgs[s["cls"]].add(s["tpg"])
        n_teams = len({(s["tpg"], s["team"]) for s in called})
        n_tpgs = len({s["tpg"] for s in called})
        print("\n  REACH (called functions only)")
        for c in (STATIC_GUARD, STATIC_FULL, SAME_OPERAND):
            print(f"  {c:14} in {len(teams[c]):5} / {n_teams} teams with calls "
                  f"({len(teams[c]) / max(1, n_teams):.1%}),  {len(tpgs[c]):4} / {n_tpgs} TPGs")

        if self.latency:
            self._print_validation()

    def _print_validation(self) -> None:
        """Latency of teams with exactly one call of an argument-sensitive
        function and no other call, split by the class of that call."""
        per_team = defaultdict(list)
        for s in self.sites:
            if s["func"] in CALLED:
                per_team[(s["tpg"], s["team"])].append(s)
        print(f"\n  VALIDATION on {self.uarch}: teams whose only call is one call to f")
        print(f"  {'function':28}{'class':>14}{'teams':>7}{'median':>8}{'p10':>7}{'p90':>7}")
        groups = defaultdict(list)
        for key, ss in per_team.items():
            if len(ss) == 1 and key in self.latency:
                groups[(ss[0]["func"], ss[0]["cls"])].append(self.latency[key])
        for func in sorted({f for f, _ in groups}):
            for c in CLASSES:
                v = np.array(groups.get((func, c), []))
                if len(v):
                    print(f"  {func:28}{c:>14}{len(v):>7}{np.median(v):>8.0f}"
                          f"{np.percentile(v, 10):>7.0f}{np.percentile(v, 90):>7.0f}")
            allv = np.concatenate([np.array(groups.get((func, c), [])) for c in CLASSES])
            dyn = np.array(groups.get((func, DYNAMIC), []))
            if len(dyn) and len(dyn) < len(allv):
                print(f"  {'':28}{'mean all → dynamic only':>28}: "
                      f"{allv.mean():.0f} → {dyn.mean():.0f} cycles")


def static_args(db: "Database | str | Path", uarch: str | None = None,
                header: str | Path | None = None) -> StaticArgsReport:
    """
    Classify every call of every TPG program of ``db``.  With ``uarch``, the
    report also compares the measured latency of single-call teams per class.
    """
    from classes.database import Database

    db_name = Path(db).name if isinstance(db, (str, Path)) else "database"
    if isinstance(db, (str, Path)):
        db = Database.load(db)

    tpgs = list(db.tpgs.values())
    hdr = Path(header) if header else next(
        (h for t in tpgs if (h := _find_header(t.source_path))), None)
    if hdr is None:
        raise FileNotFoundError("fixedptc.h not found; pass --header")
    folder = _Folder(hdr)

    rep = StaticArgsReport(db_name=db_name, header=hdr, n_tpgs=len(tpgs), uarch=uarch)
    for tpg in tpgs:
        files = _codegen(tpg.source_path)
        if files is None:
            rep.no_codegen += 1
            continue
        prog_f, tpg_f = files
        programs = {}
        for name, body in _PROG_RE.findall(prog_f.read_text()):
            programs[name], bad = analyse_program(name, body, folder)
            rep.unparsed += bad
        for tid, body in _TEAM_RE.findall(tpg_f.read_text()):
            for p in _USE_RE.findall(body):
                for cs in programs.get(p, []):
                    rep.sites.append(dict(tpg=tpg.source_path, team=int(tid),
                                          func=cs.func, cls=cs.cls, dead=cs.dead))

    if uarch:
        for tpg, tid, _ct, meas in db.get_measurements_for_uarch(uarch):
            rep.latency[(tpg.source_path, tid)] = meas.latency
    return rep
