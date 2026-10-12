"""
analysis/inventory.py — per-database inventory of TPG operations, calls and
branches.

For every ISA of a Database, one call answers:

  * **C operations** — which TPG instructions the programs use: function calls
    (``fixedpt_exp_wdiv32``, ``expf``, ``protected_fixedpt_div_32`` …) and
    operators (``X + X``, ``(X > X) ? X : X`` …), how many times they occur
    in the team code once programs are expanded into their teams, how many of
    them appear as a ``jal`` in the team code and how many do not (inlined,
    or computed at compile time — see ``TeamFacts.without_jal``), and how
    many branches each occurrence without a ``jal`` brings with it.
  * **asm operations** — every operation of the team code (the same names as
    the ``<op>_count`` features), with its count and whether the CV32E40X
    cycle table knows it.
  * **calls** — every ``jal`` target found in team code, including compiler
    helpers (``__divsi3``, ``__addsf3`` …) that have no counterpart in the C.
  * **branches** — how many branches the team code contains, where they land
    (inside the team's own address slice, after it, or inside another team's
    slice) and how far they jump.
  * **teams** — the same facts per team, to join with measurements.

Where the data comes from
-------------------------
The asm side is read from the CompiledTeams stored in the database, so it
works on any database.  The C side is read from the GEGELATI codegen next to
each TPG (``outLogs/codegen_TeamsInstrumented``, falling back to
``outLogs/codegen``).  TPGs whose codegen is missing are counted and only
drop out of the C-level columns.

How branches are attributed to C operations
-------------------------------------------
A team's code is its programs' operations laid end to end: loads, the
operations themselves, stores.  The only source of branches is therefore the
inlined code of the operations — ``+``, ``-`` and ``fixedpt_mul`` compile
branch-free, and ``(a > b) ? a : b`` becomes a single Zbb ``max``.  So, per
team::

    branches(team) ≈ Σ_op  b_op × occurrences_without_jal(op, team)

which is solved for the ``b_op`` by non-negative least squares over every
team of the ISA.  ``b_op`` is the number of branches one inlined occurrence
of ``op`` brings (a called one leaves its branches in the callee); an
operator that does branch (a ternary on a core without Zbb, say) gets
``b_op > 0`` from the same fit.  The fit's R² is reported: a low value
means branches come from somewhere the model does not see.

Usage
-----
    from analysis.inventory import inventory

    for isa, inv in inventory("db_tpg_fixedpt_iset32").items():
        inv.print()
        inv.to_csv(f"inventory/{isa}")

or ``python main.py inventory --db <path> [--isa <name>] [--csv <dir>]``.
"""

from __future__ import annotations

import bisect
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from analysis.disassembler import INSTRUCTION_LATENCY

if TYPE_CHECKING:
    from classes.database import Database


# ---------------------------------------------------------------------------
# asm parsing
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^([0-9a-f]+) <T(\d+)_(start|end)>:\s*$")
# "    33e0:\t18070e63 \tbeqz\ta4,357c <T11_start+0x1b4>"
_ASM_RE = re.compile(
    r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*([^<#]*?)\s*(?:<([^>]+)>)?\s*(?:#.*)?$"
)
_TARGET_RE = re.compile(r"(?:^|,)\s*([0-9a-f]+)\s*$")

# Conditional branches and their pseudo forms, plus the unconditional "j".
# Compressed forms ("c.beqz") are folded onto their base mnemonic.
_BRANCHES = frozenset({
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "bgt", "ble", "bgtu", "bleu",
    "j",
})
_CALLS = frozenset({"jal", "jalr", "call", "tail"})

# Landing categories for a branch leaving a team.
IN_SLICE     = "stays in its team slice"
OUT_OF_SLICE = "leaves its slice (not into another team)"
OTHER_TEAM   = "lands in another team's slice"


def _mnemonic(raw: str) -> str:
    m = raw.lower()
    return m[2:] if m.startswith("c.") else m


def _callee(sym: str | None) -> str:
    """Bare callee name: drops "+0x.." offsets and GCC clone suffixes."""
    if not sym:
        return "<indirect>"
    return sym.split("+", 1)[0].split(".", 1)[0]


@dataclass
class _Slice:
    """Address range and instruction addresses of one team's block."""
    start: int
    end: int
    addrs: list[int]


def _parse_slice(code: str) -> tuple[_Slice | None, list[dict]]:
    """Slice bounds and the instruction lines of one CompiledTeam.code."""
    start = end = None
    lines: list[dict] = []
    for line in code.splitlines():
        lm = _LABEL_RE.match(line)
        if lm:
            if lm.group(3) == "start":
                start = int(lm.group(1), 16)
            else:
                end = int(lm.group(1), 16)
            continue
        am = _ASM_RE.match(line)
        if am:
            lines.append(dict(addr=int(am.group(1), 16), mn=_mnemonic(am.group(2)),
                              ops=am.group(3), sym=am.group(4)))
    if start is None or end is None:
        return None, lines
    return _Slice(start, end, [l["addr"] for l in lines]), lines


# ---------------------------------------------------------------------------
# C codegen parsing
# ---------------------------------------------------------------------------

_PROG_RE   = re.compile(r"\b(P\d+)\s*\([^)]*\)\s*\{(.*?)\n\}", re.S)
_OPLINE_RE = re.compile(r"\breg\[\d+\]\s*=\s*(.+?);")
_CALL_RE   = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")
_TEAM_RE   = re.compile(r"\bL_T(\d+):\s*\{(.*?)\bgoto\b", re.S)
_USE_RE    = re.compile(r"=\s*(P\d+)\s*\(")


def c_operation(expr: str) -> str:
    """
    Name of the TPG instruction a program line performs.

    A call keeps its function name (``fixedpt_exp_wdiv32(op0)`` →
    ``fixedpt_exp_wdiv32``).  An operator expression is normalised so that
    every occurrence of the same operator gets the same name: operands
    (``op0``, ``reg[3]``, ``in2[1]``) become ``X`` and literals ``C``, so
    ``op0 + in2[1]`` → ``X + X``.
    """
    m = _CALL_RE.match(expr)
    if m:
        return m.group(1)
    e = re.sub(r"\bop\d+\b|\breg\[\d+\]|\bin\d+\[\d+\]", "X", expr)
    e = re.sub(r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[fFuUlL]*\b", "C", e)
    return re.sub(r"\s+", " ", e).strip()


def c_sites_per_team(source_path: str) -> dict[int, Counter] | None:
    """
    ``{team_id: Counter(operation -> occurrences)}`` from a TPG's codegen.

    Programs are ``always_inline``, so a program used by three teams puts its
    operations three times into the binary: occurrences are counted per team,
    after that expansion.  Returns None when no codegen is found.
    """
    root = Path(source_path) / "outLogs"
    for sub in ("codegen_TeamsInstrumented", "codegen"):
        prog_f, tpg_f = root / sub / "TPG_program.h", root / sub / "TPG.c"
        if prog_f.exists() and tpg_f.exists():
            break
    else:
        return None

    # "fixedpt reg[8] = {0, ...};" initialises the registers; it is not an
    # operation, and its brace initialiser is what tells it apart.
    per_prog = {
        name: Counter(c_operation(e) for e in _OPLINE_RE.findall(body)
                      if not e.lstrip().startswith("{"))
        for name, body in _PROG_RE.findall(prog_f.read_text())
    }
    teams: dict[int, Counter] = {}
    for tid, body in _TEAM_RE.findall(tpg_f.read_text()):
        sites: Counter = Counter()
        for p in _USE_RE.findall(body):
            sites.update(per_prog.get(p, {}))
        teams[int(tid)] = sites
    return teams


# ---------------------------------------------------------------------------
# Per-team facts
# ---------------------------------------------------------------------------

@dataclass
class TeamFacts:
    """Everything the inventory knows about one CompiledTeam."""
    tpg: str
    team_id: int
    n_instructions: int
    asm_ops: Counter
    calls: Counter                   # callee -> jal count, in the team slice
    branches: list[dict]             # kind, mnemonic, distance
    c_sites: Counter | None          # C operation -> occurrences (None: no codegen)

    def without_jal(self, op: str) -> int:
        """
        C occurrences of ``op`` with no matching jal in the team slice.

        Two causes, both checked on the fixedpt and float databases:

        * inlining — the compiler copied the body of a ``static inline``
          function (fixedpt code) into the team;
        * constant folding — the operand is a register the program never
          writes, so it is always 0 and the compiler computes the result at
          compile time: ``expf(reg[3])`` is ``expf(0.0f) = 1.0f``, no call
          emitted.  ``logf(0)`` is the exception: it is a pole error, so the
          call is kept.
        """
        return max(0, self.c_sites.get(op, 0) - self.calls.get(op, 0))


def _team_facts(tpg, team_id: int, ct, slc: _Slice, lines: list[dict],
                all_slices: list[tuple[int, _Slice]], c_sites) -> TeamFacts:
    branches, calls = [], Counter()
    for l in lines:
        if l["mn"] in _CALLS:
            calls[_callee(l["sym"])] += 1
            continue
        if l["mn"] not in _BRANCHES:
            continue
        tm = _TARGET_RE.search(l["ops"])
        if not tm:
            continue
        tgt = int(tm.group(1), 16)
        if slc.start <= tgt < slc.end:
            kind = IN_SLICE
            # exact distance in instructions, robust to 2-byte encodings
            dist = bisect.bisect_left(slc.addrs, tgt) - bisect.bisect_left(slc.addrs, l["addr"])
        else:
            kind = OUT_OF_SLICE
            for other_id, other in all_slices:
                if other_id != team_id and other.start <= tgt < other.end:
                    kind = OTHER_TEAM
                    break
            dist = None
        branches.append(dict(kind=kind, mn=l["mn"], dist=dist,
                             bytes=tgt - l["addr"]))
    return TeamFacts(
        tpg=tpg.name, team_id=team_id,
        n_instructions=len(ct.instructions or []),
        asm_ops=Counter(i.operation for i in (ct.instructions or [])),
        calls=calls, branches=branches, c_sites=c_sites,
    )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class Inventory:
    """
    Inventory of one ISA of one database.  Every table is a list of plain
    dicts, so it prints, writes to CSV, or loads into any dataframe library.
    """
    db_name: str
    isa: str
    n_tpgs: int
    n_teams: int
    n_tpgs_without_codegen: int
    c_ops: list[dict] = field(default_factory=list)
    asm_ops: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    branches: list[dict] = field(default_factory=list)
    teams: list[dict] = field(default_factory=list)
    fit_r2: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    def print(self, top: int = 40) -> None:
        bar = "─" * 96
        print(f"\n{bar}\n  INVENTORY  {self.db_name}  |  ISA {self.isa}\n{bar}")
        print(f"  {self.n_tpgs} TPGs, {self.n_teams} teams"
              + (f"  ({self.n_tpgs_without_codegen} TPGs without codegen: "
                 "excluded from the C-level tables)" if self.n_tpgs_without_codegen else ""))

        print("\n  C OPERATIONS (programs expanded into their teams)")
        print(f"  {'operation':<30}{'TPGs':>6}{'teams':>7}{'occur.':>8}{'jal':>6}"
              f"{'no jal':>8}{'br/occ':>8}{'  in':>6}{' out':>6}")
        for r in self.c_ops[:top]:
            print(f"  {r['operation'][:29]:<30}{r['tpgs']:>6}{r['teams']:>7}"
                  f"{r['occurrences']:>8}{r['jal']:>6}{r['without_jal']:>8}"
                  f"{r['branches_per_occurrence']:>8.2f}{r['branches_in_slice']:>6.2f}"
                  f"{r['branches_out_of_slice']:>6.2f}")
        print("  no jal : occurrences with no jal in the team slice: inlined, or computed")
        print("           at compile time because the operand is an unwritten register (= 0)")
        print("  br/occ : branches brought by one such occurrence (split in / out of slice)")
        if self.fit_r2:
            fmt = lambda v: "n/a" if v != v else f"{v:.3f}"     # NaN: no branch at all
            print(f"  branch attribution fit R²: total {fmt(self.fit_r2['total'])}, "
                  f"in-slice {fmt(self.fit_r2['in_slice'])}, "
                  f"out-of-slice {fmt(self.fit_r2['out_of_slice'])}")

        print("\n  CALLS (jal targets inside team slices)")
        if not self.calls:
            print("  none")
        for r in self.calls:
            print(f"  {r['callee']:<32}{r['jal']:>6} jal in {r['teams']:>5} teams   "
                  f"{r['origin']}")

        print("\n  BRANCHES")
        for r in self.branches:
            print(f"  {r['metric']:<52}{r['value']}")

        print(f"\n  ASM OPERATIONS (top {top} of {len(self.asm_ops)})")
        print(f"  {'operation':<30}{'count':>8}{'teams':>7}{'TPGs':>6}  class      in doc table")
        for r in self.asm_ops[:top]:
            print(f"  {r['operation'][:29]:<30}{r['count']:>8}{r['teams']:>7}{r['tpgs']:>6}"
                  f"  {r['class']:<10} {'yes' if r['in_doc_table'] else 'NO'}")
        missing = [r["operation"] for r in self.asm_ops if not r["in_doc_table"]]
        if missing:
            print(f"  not in the CV32E40X cycle table: {', '.join(missing)}")
        print(bar)

    def to_csv(self, prefix: str | Path) -> list[Path]:
        """Write one CSV per table as ``<prefix>_<table>.csv``."""
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for name in ("c_ops", "asm_ops", "calls", "branches", "teams"):
            rows = getattr(self, name)
            if not rows:
                continue
            path = prefix.parent / f"{prefix.name}_{name}.csv"
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            written.append(path)
        return written


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss else float("nan")


def _build(db_name: str, isa: str, facts: list[TeamFacts], n_tpgs: int,
           n_no_codegen: int) -> Inventory:
    from scipy.optimize import nnls

    inv = Inventory(db_name=db_name, isa=isa, n_tpgs=n_tpgs, n_teams=len(facts),
                    n_tpgs_without_codegen=n_no_codegen)
    with_c = [f for f in facts if f.c_sites is not None]
    c_names = sorted({op for f in with_c for op in f.c_sites})

    # ── C operations + branch attribution ─────────────────────────────
    per_op: dict[str, np.ndarray] = {}
    if with_c and c_names:
        A = np.array([[f.without_jal(op) for op in c_names] for f in with_c], float)
        y = {k: np.array([sum(1 for b in f.branches if sel(b)) for f in with_c], float)
             for k, sel in (("total", lambda b: True),
                            ("in_slice", lambda b: b["kind"] == IN_SLICE),
                            ("out_of_slice", lambda b: b["kind"] != IN_SLICE))}
        for k, target in y.items():
            coef, _ = nnls(A, target)
            per_op[k] = coef
            inv.fit_r2[k] = _r2(target, A @ coef)

    for j, op in enumerate(c_names):
        teams_with = [f for f in with_c if f.c_sites.get(op)]
        occ = sum(f.c_sites[op] for f in teams_with)
        jal = sum(min(f.calls.get(op, 0), f.c_sites[op]) for f in teams_with)
        inv.c_ops.append(dict(
            operation=op,
            kind=("call" if re.fullmatch(r"[A-Za-z_]\w*", op) and op not in ("X", "C")
                  else "operator"),
            tpgs=len({f.tpg for f in teams_with}),
            teams=len(teams_with),
            occurrences=occ,
            jal=jal,
            without_jal=occ - jal,
            branches_per_occurrence=round(float(per_op["total"][j]), 3) if per_op else 0.0,
            branches_in_slice=round(float(per_op["in_slice"][j]), 3) if per_op else 0.0,
            branches_out_of_slice=round(float(per_op["out_of_slice"][j]), 3) if per_op else 0.0,
        ))
    inv.c_ops.sort(key=lambda r: -r["occurrences"])

    # ── calls ─────────────────────────────────────────────────────────
    known = set(c_names)
    callee_count, callee_teams = Counter(), Counter()
    for f in facts:
        callee_count.update(f.calls)
        callee_teams.update(set(f.calls))
    for callee, n in callee_count.most_common():
        inv.calls.append(dict(
            callee=callee, jal=n, teams=callee_teams[callee],
            origin=("called by the programs" if callee in known
                    else "compiler helper, no C counterpart" if callee.startswith("__")
                    else "not called directly by any program (nested call?)"),
        ))

    # ── branches ──────────────────────────────────────────────────────
    allb = [b for f in facts for b in f.branches]
    kinds = Counter(b["kind"] for b in allb)
    n = len(allb)
    rows = [("teams", len(facts)),
            ("teams with at least one branch", sum(1 for f in facts if f.branches)),
            ("branches in team slices", n)]
    for kind in (IN_SLICE, OUT_OF_SLICE, OTHER_TEAM):
        rows.append((f"  {kind}", f"{kinds[kind]} ({100 * kinds[kind] / n:.1f}%)" if n else 0))
    d_in = np.array([b["dist"] for b in allb if b["kind"] == IN_SLICE], float)
    if len(d_in):
        fwd = d_in[d_in > 0]
        rows.append(("in-slice jump, instructions: median / p90",
                     f"{np.median(d_in):+.0f} / {np.percentile(d_in, 90):+.0f}"))
        rows.append(("in-slice backward jumps (loops)", int((d_in < 0).sum())))
        for lo, hi in ((1, 3), (3, 6), (6, 11), (11, 26), (26, 51), (51, None)):
            m = (fwd >= lo) & (fwd < hi) if hi else (fwd >= lo)
            label = f"{lo}-{hi - 1}" if hi else f"{lo}+"
            rows.append((f"  forward, skips {label} instr",
                         f"{int(m.sum())} ({100 * m.mean():.1f}%)" if len(fwd) else 0))
    d_out = np.array([b["bytes"] for b in allb if b["kind"] != IN_SLICE], float)
    if len(d_out):
        rows.append(("out-of-slice jump, bytes: median / p90",
                     f"{np.median(d_out):+.0f} / {np.percentile(d_out, 90):+.0f}"))
    for mn, c in Counter(b["mn"] for b in allb).most_common():
        out = sum(1 for b in allb if b["mn"] == mn and b["kind"] != IN_SLICE)
        rows.append((f"  {mn}", f"{c}  (leaving the slice: {out}, {100 * out / c:.0f}%)"))
    inv.branches = [dict(metric=k, value=v) for k, v in rows]

    # ── asm operations ────────────────────────────────────────────────
    op_count, op_teams, op_tpgs = Counter(), Counter(), defaultdict(set)
    for f in facts:
        op_count.update(f.asm_ops)
        op_teams.update(set(f.asm_ops))
        for op in f.asm_ops:
            op_tpgs[op].add(f.tpg)
    for op, c in op_count.most_common():
        base = op.split("_", 1)[0]
        inv.asm_ops.append(dict(
            operation=op, count=c, teams=op_teams[op], tpgs=len(op_tpgs[op]),
            **{"class": "call" if base in _CALLS else
                        "branch" if _mnemonic(base) in _BRANCHES else "other"},
            in_doc_table=op.lower() in INSTRUCTION_LATENCY or base.lower() in INSTRUCTION_LATENCY,
        ))

    # ── per team ──────────────────────────────────────────────────────
    for f in facts:
        k = Counter(b["kind"] for b in f.branches)
        inv.teams.append(dict(
            tpg=f.tpg, team_id=f.team_id, instructions=f.n_instructions,
            branches_in_slice=k[IN_SLICE], branches_out_of_slice=k[OUT_OF_SLICE],
            branches_other_team=k[OTHER_TEAM],
            jal=";".join(f"{c}:{n}" for c, n in sorted(f.calls.items())),
            c_operations=("" if f.c_sites is None else
                          ";".join(f"{o}:{n}" for o, n in sorted(f.c_sites.items()))),
        ))
    return inv


def inventory(db: "Database | str | Path", isa: str | None = None) -> dict[str, Inventory]:
    """
    Build the inventory of every ISA of ``db`` (or of ``isa`` only).

    ``db`` is a Database or the path of a pickled one.  Returns
    ``{isa_name: Inventory}``.
    """
    from classes.database import Database

    db_name = Path(db).name if isinstance(db, (str, Path)) else "database"
    if isinstance(db, (str, Path)):
        db = Database.load(db)

    # Group every CompiledTeam by ISA, then by TPG so that each team can be
    # compared with the slices of the other teams of the same binary.
    by_isa: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    tpgs = {}
    for tpg, team_id, ct in db.all_compiled_teams():
        if isa is not None and ct.isa_name != isa:
            continue
        by_isa[ct.isa_name][tpg.source_path].append((team_id, ct))
        tpgs[tpg.source_path] = tpg

    out: dict[str, Inventory] = {}
    for isa_name, per_tpg in sorted(by_isa.items()):
        facts: list[TeamFacts] = []
        no_codegen = 0
        for src, teams in per_tpg.items():
            c_teams = c_sites_per_team(src)
            no_codegen += c_teams is None
            parsed = [(tid, ct, *_parse_slice(ct.code or "")) for tid, ct in teams]
            slices = [(tid, s) for tid, _ct, s, _l in parsed if s is not None]
            for tid, ct, slc, lines in parsed:
                if slc is None:
                    continue
                facts.append(_team_facts(
                    tpgs[src], tid, ct, slc, lines, slices,
                    None if c_teams is None else c_teams.get(tid, Counter())))
        out[isa_name] = _build(db_name, isa_name, facts, len(per_tpg), no_codegen)
    return out
