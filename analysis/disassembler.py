from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from classes.instruction import Instruction
from classes.features import FeatureVector


# ---------------------------------------------------------------------------
# Table-based latencies (CV32E40X)
#
# Source: OpenHW Group CV32E40X user manual, "Pipeline Details" → table
# "Cycle Counts Per Instruction Type" (docs/user_manual/source/pipeline.rst).
# Every number below is a row of that table.
#
# What the table does NOT cover
# -----------------------------
# 1. Hazards.  The manual lists those separately, as penalties on *pairs* of
#    instructions: +1 for a load-use hazard, +1 for jalr depending on a
#    non-load, +2 for jalr depending on a load, +1/+2 for the various CSR
#    read/write conflicts.  They are not per-instruction costs, so they are
#    not in this table — they are what the raw_d<N>_<producer>_<consumer>
#    features of FeaturesAnalyzer exist to capture.  A sum over this table is
#    therefore a floor: what a stall-free run of the block would cost.
#
# 2. Callees.  "jal <callee>" costs the 2 cycles of the jump itself; the
#    cycles spent inside the callee are not modelled.  See CALL_BODY_CYCLES.
#
# Run-time-dependent rows
# -----------------------
# Three rows depend on state a disassembly cannot show.  Each gets a named
# constant so the assumption is visible and changeable in one place:
# division latency (divisor-dependent), branch cost (taken or not), and
# load/store cost (aligned or not).
#
# A fourth dependency does not apply here: the "+1 cycle for a
# non-word-aligned non-RVC target" variants of the jump/branch/fence rows can
# only occur when the C extension makes 2-byte-aligned targets possible.  The
# uarch this database targets (cv32e40x_im1_zba_zbb, ISA rv32im_zicsr_zba_zbb)
# has no C extension, so every target is word-aligned and the base cost always
# applies.
# ---------------------------------------------------------------------------

# -- Rows that are a single number ------------------------------------------
INTEGER_LATENCY   = 1   # "Integer Computational": 1
MUL_CYCLES        = 1   # "Multiplication": 1 for mul …
MULH_CYCLES       = 4   #                   … 4 for mulh/mulhsu/mulhu
JUMP_CYCLES       = 2   # "Jump": 2 (jumps resolve in ID, flushing IF)
MRET_CYCLES       = 2   # "mret": 2
FENCE_CYCLES      = 5   # "fence" / "fence.i": 5 (implemented as a jump)
CSR_CYCLES        = 1   # "CSR Access": 1 (4 only for jvt, a Zcmt CSR)
BITMANIP_CYCLES   = 1   # "Zba, Zbb, Zbc, Zbs": 1

# -- Rows that depend on run-time state -------------------------------------
# "Division/Remainder": 3–35, "depends on the divider operand value
# (operand b)" — the divider iterates over the significant bits of the
# divisor.  The disassembly cannot know that value, so the worst case is used;
# set this to DIV_CYCLES_MIN, or to a measured average, to model it otherwise.
DIV_CYCLES_MIN = 3
DIV_CYCLES_MAX = 35
DIV_CYCLES     = DIV_CYCLES_MAX

# "Branch (Not-Taken)": 1, "Branch (Taken)": 3.  Static code gives one entry
# per branch with no way to know how often it is taken; the not-taken cost is
# used because it is the one that "will not stall".
BRANCH_NOT_TAKEN_CYCLES = 1
BRANCH_TAKEN_CYCLES     = 3
BRANCH_CYCLES           = BRANCH_NOT_TAKEN_CYCLES

# "Load/Store": 1, or 2 for a non-word-aligned word transfer or a halfword
# crossing a word boundary.  Alignment is a run-time address property, so the
# aligned cost is used.
LOAD_STORE_CYCLES            = 1
LOAD_STORE_MISALIGNED_CYCLES = 2

# Cycles spent *inside* a callee, keyed by callee symbol as Instruction
# .operation spells it after the mnemonic ("fixedpt_ln_wdiv32" for
# "jal_fixedpt_ln_wdiv32").  Left empty: a call costs only its jump until the
# callee is itself measured.  Fill it to fold a measured function cost into
# the estimate.
CALL_BODY_CYCLES: dict[str, int] = {}


# -- The table, grouped by the manual's rows --------------------------------

# "Integer Computational" — RV32I, 1 cycle.  Listed explicitly rather than
# left to the default so that an unknown mnemonic stays distinguishable from
# a known 1-cycle one.
_INTEGER_COMPUTATIONAL = {
    "lui", "auipc",
    "addi", "slti", "sltiu", "xori", "ori", "andi", "slli", "srli", "srai",
    "add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
    "ecall", "ebreak",
}

_LOAD_STORE = {"lb", "lh", "lw", "lbu", "lhu", "sb", "sh", "sw"}

_BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}

# Zba / Zbb / Zbc / Zbs, all 1 cycle.  The uarch here implements Zba+Zbb only;
# Zbc/Zbs are kept so the table stays valid for a differently-configured core.
_BITMANIP = {
    # Zba
    "sh1add", "sh2add", "sh3add",
    # Zbb
    "andn", "orn", "xnor", "clz", "ctz", "cpop", "pcnt",
    "max", "maxu", "min", "minu", "sext.b", "sext.h", "zext.h",
    "rol", "ror", "rori", "rev8", "orc.b",
    # Zbc
    "clmul", "clmulh", "clmulr",
    # Zbs
    "bclr", "bclri", "bext", "bexti", "binv", "binvi", "bset", "bseti",
}

_CSR = {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"}

# Pseudo-instructions.  objdump prints these instead of the base instruction
# they alias, so they must be costed or every one of them would fall through
# to the unknown-mnemonic default.  Each takes the cost of its expansion.
_PSEUDO: dict[str, int] = {
    "nop":   INTEGER_LATENCY,   # addi x0, x0, 0
    "mv":    INTEGER_LATENCY,   # addi rd, rs, 0
    "li":    INTEGER_LATENCY,   # addi rd, x0, imm   (a wide li is printed as
                                # lui + addi, i.e. two separate lines)
    "not":   INTEGER_LATENCY,   # xori rd, rs, -1
    "neg":   INTEGER_LATENCY,   # sub  rd, x0, rs
    "seqz":  INTEGER_LATENCY,   # sltiu rd, rs, 1
    "snez":  INTEGER_LATENCY,   # sltu rd, x0, rs
    "sltz":  INTEGER_LATENCY,   # slt  rd, rs, x0
    "sgtz":  INTEGER_LATENCY,   # slt  rd, x0, rs
    "beqz":  BRANCH_CYCLES,     # beq  rs, x0, off
    "bnez":  BRANCH_CYCLES,     # bne  rs, x0, off
    "blez":  BRANCH_CYCLES,     # bge  x0, rs, off
    "bgez":  BRANCH_CYCLES,     # bge  rs, x0, off
    "bltz":  BRANCH_CYCLES,     # blt  rs, x0, off
    "bgtz":  BRANCH_CYCLES,     # blt  x0, rs, off
    "bgt":   BRANCH_CYCLES,     # blt  rt, rs, off
    "ble":   BRANCH_CYCLES,     # bge  rt, rs, off
    "bgtu":  BRANCH_CYCLES,     # bltu rt, rs, off
    "bleu":  BRANCH_CYCLES,     # bgeu rt, rs, off
    "j":     JUMP_CYCLES,       # jal  x0, off
    "jr":    JUMP_CYCLES,       # jalr x0, rs, 0
    "ret":   JUMP_CYCLES,       # jalr x0, ra, 0
    "call":  INTEGER_LATENCY + JUMP_CYCLES,   # auipc + jalr, printed as one
    "tail":  INTEGER_LATENCY + JUMP_CYCLES,   # auipc + jalr, printed as one
    "csrr":  CSR_CYCLES,        # csrrs rd, csr, x0
    "csrw":  CSR_CYCLES,        # csrrw x0, csr, rs
    "csrs":  CSR_CYCLES,        # csrrs x0, csr, rs
    "csrc":  CSR_CYCLES,        # csrrc x0, csr, rs
    "csrwi": CSR_CYCLES,        # csrrwi x0, csr, imm
    "csrsi": CSR_CYCLES,        # csrrsi x0, csr, imm
    "csrci": CSR_CYCLES,        # csrrci x0, csr, imm
    "fence.i": FENCE_CYCLES,    # Zifencei, same row as fence
}

INSTRUCTION_LATENCY: dict[str, int] = {
    **{m: INTEGER_LATENCY   for m in _INTEGER_COMPUTATIONAL},
    **{m: LOAD_STORE_CYCLES for m in _LOAD_STORE},
    **{m: BRANCH_CYCLES     for m in _BRANCH},
    **{m: BITMANIP_CYCLES   for m in _BITMANIP},
    **{m: CSR_CYCLES        for m in _CSR},
    "mul": MUL_CYCLES, "mulh": MULH_CYCLES,
    "mulhsu": MULH_CYCLES, "mulhu": MULH_CYCLES,
    "div": DIV_CYCLES, "divu": DIV_CYCLES,
    "rem": DIV_CYCLES, "remu": DIV_CYCLES,
    "jal": JUMP_CYCLES, "jalr": JUMP_CYCLES,
    "mret": MRET_CYCLES,
    "fence": FENCE_CYCLES,
    **_PSEUDO,
}

# Mnemonics already reported as missing from the table, so the warning is
# printed once per mnemonic instead of once per occurrence.
_WARNED_UNKNOWN: set[str] = set()

# Regex matching one disassembled line:
#   <addr>:  <machine-code>   <mnemonic> [operands]   [<label>]   [# comment]
#
# objdump appends two kinds of trailing annotation that are NOT operands:
#
#   1. Bare label:   "blez a7,886 <T2_start+0x1e>"
#      Captured by  (?:\s+(<\S+>))?   before the optional # comment.
#
#   2. Hash comment: "lw a7,4(a3) # 20000004 <_sp+…>"
#      Captured by  (?:\s*#.*)?
#
# The non-greedy (.+?) stops before either optional suffix, so group 1
# always contains only mnemonic + clean operands.
#
# Group 2 is the bare label, kept rather than discarded: for a call
# ("jal ra,884 <expf>") it names the callee, which Instruction folds into
# its `operation` so that jal expf / jal logf / jal sinf are distinct
# operations.  For every other instruction it is still ignored.
_LINE_RE = re.compile(
    r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(.+?)(?:\s+(<\S+>))?(?:\s*#.*)?$"
)


def _instruction_from_match(m: "re.Match[str]") -> Instruction:
    """Build an Instruction from a _LINE_RE match, keeping the call symbol."""
    return Instruction.parse(m.group(1).strip(), symbol=m.group(2))

# Matches an exact team label line, e.g. "000007c2 <T0_start>:"
_TEAM_LABEL_RE = re.compile(r"^[0-9a-f]{8} <T(\d+)_(start|end)>:\s*$")

# The first instruction inside T_start is always "csrr <rd>,mcycle" —
# a timing probe inserted by the test harness. Strip it from code & instrs.
_MCYCLE_PROBE_RE = re.compile(
    r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+csrr\s+\w+,mcycle"
)


@dataclass
class TeamBlock:
    """
    Everything extracted for one team from the disassembly file.

    Attributes
    ----------
    team_id:      Integer from the T<N>_start label.
    instructions: Ordered list of Instructions (timing probe excluded).
    code:         Raw disassembly text of the block — the label lines plus
                  every instruction line exactly as printed by objdump,
                  with the ``csrr mcycle`` timing probe line stripped.
    """
    team_id: int
    instructions: list[Instruction]
    code: str


@dataclass
class TeamLatency:
    """Measured latency statistics for a single team from the JSON results."""
    team_id: int
    nb_measurements: int       # Count in the JSON — number of individual runs
    avg_cycles: float          # AvgCyclesPerTeam
    stddev_cycles: float       # StddevCyclesPerTeam
    coefficient_variation: float


@dataclass
class ClassLatency:
    """
    Whole-TPG latency for one *class* (a.k.a. graph traversal / learning
    example class) reported in a JSON "Classes" array.

    A class corresponds to one path through the TPG — the same integer id
    used by the ``[k] -> [T.. -> T..]`` mapping in ``LE_states.h``.

    Attributes
    ----------
    class_id:              The "Class" field (== traversal id in LE_states.h).
    nb_measurements:       "Count" — number of individual runs averaged.
    avg_cycles:            "AvgCyclesPerClass".
    stddev_cycles:         "StddevCyclesPerClass".
    coefficient_variation: "CoefficientVariation".
    """
    class_id: int
    nb_measurements: int
    avg_cycles: float
    stddev_cycles: float
    coefficient_variation: float


@dataclass
class DispatchLatency:
    """
    One entry of the ``Dispatches`` array — the aggregate cost of every
    dispatch of a given size in one TPG.

    The dispatch is the control step following a Team's execution: the
    programs' results are compared and the winner selects the next
    destination (Team or Action).  It is instrumented as a whole (via the
    ``dispatch_start`` / ``dispatch_end`` probes in ``inferenceTPG``) and
    reported per ``DispatchSize``, with no instruction-level breakdown.

    Attributes
    ----------
    dispatch_size:         The "DispatchSize" field — number of programs compared.
    nb_measurements:       "Count" — number of dispatch events averaged.
    avg_cycles:            "AvgCyclesPerDispatch".
    stddev_cycles:         "StddevCyclesPerDispatch".
    coefficient_variation: "CoefficientVariation".
    """
    dispatch_size: int
    nb_measurements: int
    avg_cycles: float
    stddev_cycles: float
    coefficient_variation: float


@dataclass
class TPGLatencyData:
    """
    All latency information parsed from one JSON results file.

    The JSON contains two whole-TPG benchmarks:

      * ``instrTPG``            — only the TPG is instrumented.
      * ``instrTeams_instrTPG`` — the TPG *and* every team are instrumented;
                                  the extra per-team timing probes add an
                                  instrumentation *overcost* on top of the
                                  ``instrTPG`` figure.

    Both are captured per class so the overcost can be quantified traversal
    by traversal.  Per-team latencies come from the ``Teams`` array of the
    ``instrTeams_instrTPG`` section (teams are only measurable when they are
    instrumented).

    Attributes
    ----------
    simulator:          Simulator / uarch name  (e.g. "cv32e40x_im2_zba_zbb").
    isa:                ISA string              (e.g. "rv32ic_zicsr_zmmul_zba_zbb").
    abi:                ABI string              (e.g. "ilp32").
    dtype:              Data type               (e.g. "fixedpt").
    tpg_mean_lat:       Mean whole-TPG latency, teams+TPG instrumented (cycles).
    tpg_stddev_lat:     Stddev of the above.
    tpg_only_mean_lat:  Mean whole-TPG latency, TPG-only instrumented (cycles).
    tpg_only_stddev_lat:Stddev of the above.
    team_latencies:     Dict team_id → TeamLatency (instrTeams_instrTPG.Teams).
    classes_tpg_only:   Dict class_id → ClassLatency from ``instrTPG``.
    classes_tpg_teams:  Dict class_id → ClassLatency from ``instrTeams_instrTPG``.
    dispatches:         Dict dispatch_size → DispatchLatency, from the
                        ``Dispatches`` array of
                        ``instrDispatch_instrTeams_instrTPG``.  Empty when the
                        build was not dispatch-instrumented.
    has_dispatch_data:  Whether that section was present at all.
    """
    simulator: str
    isa: str
    abi: str
    dtype: str
    tpg_mean_lat: float
    tpg_stddev_lat: float
    team_latencies: dict[int, TeamLatency] = field(default_factory=dict)
    tpg_only_mean_lat: float = 0.0
    tpg_only_stddev_lat: float = 0.0
    classes_tpg_only:  dict[int, ClassLatency] = field(default_factory=dict)
    classes_tpg_teams: dict[int, ClassLatency] = field(default_factory=dict)
    dispatches:        dict[int, DispatchLatency] = field(default_factory=dict)
    has_dispatch_data: bool = False

    def get_dispatch_latency(self, dispatch_size: int) -> "DispatchLatency | None":
        """Return the DispatchLatency for *dispatch_size*, or None."""
        return self.dispatches.get(dispatch_size)

    def get_team_latency(self, team_id: int) -> TeamLatency | None:
        """Return the TeamLatency for *team_id*, or None if not measured."""
        return self.team_latencies.get(team_id)


class Disassembler:
    """
    Parses raw RISC-V assembly text (objdump -d output) into Instruction objects.

    All methods are static — no instance state needed.
    """

    # ------------------------------------------------------------------ #
    # Public API — single-block helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_assembly(code: str) -> list[Instruction]:
        """
        Parse a multi-line assembly string and return one Instruction per
        decoded line.  Comment-only lines and blank lines are silently skipped.
        """
        instructions: list[Instruction] = []
        for line in code.splitlines():
            m = _LINE_RE.search(line)
            if m:
                instructions.append(_instruction_from_match(m))
        return instructions

    @staticmethod
    def parse_file(path: str) -> list[Instruction]:
        """Convenience wrapper: read a file and call parse_assembly."""
        with open(path, "r") as fh:
            return Disassembler.parse_assembly(fh.read())

    # ------------------------------------------------------------------ #
    # Public API — TPG-aware multi-team parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_tpg_file(path: str) -> dict[int, TeamBlock]:
        """
        Parse an objdump file that contains labelled team blocks::

            000007c2 <T0_start>:
                7c2:   b0002ef3    csrr  t4,mcycle   ← timing probe, stripped
                7c6:   0046a883    lw    a7,4(a3)
                …
            0000080c <T0_end>:

        The ``csrr mcycle`` timing probe (first line after each T_start label)
        is stripped from both the instruction list and the raw code text.

        Assumption, and where it breaks.  A team's block is taken to be the
        lines between its two labels, one team after the other.  The labels
        only mark basic blocks, though, and the compiler is free to lay those
        out in another order.  When the library functions are inlined into
        the teams (``inline`` builds), GCC's block reordering and jump
        threading can move the end of team n past the start of team n+1.  The
        file then reads ``T<n>_start … T<n+1>_start … T<n>_end`` and:

          * team n gets no block at all — the loader has no code for it and
            skips the team;
          * if ``T<n>_end`` also comes before ``T<n+1>_end``, the block of
            team n+1 stops there, short of its own end label.

        Both cases are reported in one warning per file; what is returned is
        unchanged.  Recovering such teams means following the control flow
        from ``T<n>_start`` to ``T<n>_end`` instead of reading a line range,
        which this parser does not do.  ``noipa`` builds keep the functions
        behind calls and do not show the problem.

        Returns:
            ``{team_id: TeamBlock}`` — one entry per team in the file.
        """
        blocks: dict[int, TeamBlock] = {}
        current_id: int | None = None
        probe_consumed: bool = False
        raw_lines: list[str] = []
        instructions: list[Instruction] = []
        # Overlapping labels: (team left without a block, team that opened
        # inside it) and (team whose block was cut, team whose end cut it).
        no_block: list[tuple[int, int]] = []
        cut_short: list[tuple[int, int]] = []

        with open(path, "r") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                label_m = _TEAM_LABEL_RE.match(line)
                if label_m:
                    tid    = int(label_m.group(1))
                    marker = label_m.group(2)

                    if marker == "start":
                        if current_id is not None:
                            no_block.append((current_id, tid))
                        current_id     = tid
                        probe_consumed = False
                        raw_lines      = [line]
                        instructions   = []
                    elif marker == "end" and current_id is not None:
                        if tid != current_id:
                            cut_short.append((current_id, tid))
                        raw_lines.append(line)
                        blocks[current_id] = TeamBlock(
                            team_id=current_id,
                            instructions=instructions,
                            code="\n".join(raw_lines),
                        )
                        current_id = None
                    continue

                if current_id is None:
                    continue   # harness glue — skip

                if not probe_consumed:
                    if _MCYCLE_PROBE_RE.match(line):
                        probe_consumed = True
                    continue   # skip probe line itself and any blank lines before it

                # Normal instruction line
                raw_lines.append(line)
                instr_m = _LINE_RE.search(line)
                if instr_m:
                    instructions.append(_instruction_from_match(instr_m))

        if no_block or cut_short:
            details = [f"no block for T{lost} (T{inner}_start comes before T{lost}_end)"
                       for lost, inner in no_block]
            details += [f"T{cut} cut short at T{by}_end"
                        for cut, by in cut_short]
            print(f"Warning: team labels overlap in {path}: " + "; ".join(details))

        return blocks

    # ------------------------------------------------------------------ #
    # Public API — JSON latency parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_latency_json(path: str) -> TPGLatencyData:
        """
        Parse a JSON results file.

        ``instrTeams_instrTPG.Teams`` supplies the per-team latencies, the two
        whole-TPG sections supply the per-class figures, and — when present —
        ``instrDispatch_instrTeams_instrTPG.Dispatches`` supplies the
        per-dispatch-size figures.
        """
        with open(path, "r") as fh:
            data: dict = json.load(fh)

        section = data["instrTeams_instrTPG"]          # TPG + teams instrumented
        tpg_only_section = data.get("instrTPG", {})    # TPG-only instrumented
        # Dispatch instrumentation is optional: older result folders have no
        # such section and simply produce no dispatch data.
        dispatch_section = data.get("instrDispatch_instrTeams_instrTPG", {})

        team_latencies: dict[int, TeamLatency] = {}
        for entry in section.get("Teams", []):
            tid = int(entry["Team"])
            team_latencies[tid] = TeamLatency(
                team_id=tid,
                nb_measurements=int(entry["Count"]),
                avg_cycles=float(entry["AvgCyclesPerTeam"]),
                stddev_cycles=float(entry["StddevCyclesPerTeam"]),
                coefficient_variation=float(entry["CoefficientVariation"]),
            )

        def _parse_classes(sec: dict) -> dict[int, ClassLatency]:
            out: dict[int, ClassLatency] = {}
            for entry in sec.get("Classes", []):
                cid = int(entry["Class"])
                out[cid] = ClassLatency(
                    class_id=cid,
                    nb_measurements=int(entry["Count"]),
                    avg_cycles=float(entry["AvgCyclesPerClass"]),
                    stddev_cycles=float(entry["StddevCyclesPerClass"]),
                    coefficient_variation=float(entry["CoefficientVariation"]),
                )
            return out

        dispatches: dict[int, DispatchLatency] = {}
        for entry in dispatch_section.get("Dispatches", []):
            size = int(entry["DispatchSize"])
            dispatches[size] = DispatchLatency(
                dispatch_size=size,
                nb_measurements=int(entry["Count"]),
                avg_cycles=float(entry["AvgCyclesPerDispatch"]),
                stddev_cycles=float(entry["StddevCyclesPerDispatch"]),
                coefficient_variation=float(entry["CoefficientVariation"]),
            )

        return TPGLatencyData(
            simulator=data["simulator"],
            isa=data["isa"],
            abi=data["abi"],
            dtype=data["dtype"],
            tpg_mean_lat=float(section["tpg_mean_lat"]),
            tpg_stddev_lat=float(section["tpg_stddev_lat"]),
            team_latencies=team_latencies,
            tpg_only_mean_lat=float(tpg_only_section.get("tpg_mean_lat", 0.0)),
            tpg_only_stddev_lat=float(tpg_only_section.get("tpg_stddev_lat", 0.0)),
            classes_tpg_only=_parse_classes(tpg_only_section),
            classes_tpg_teams=_parse_classes(section),
            dispatches=dispatches,
            has_dispatch_data=bool(dispatch_section),
        )

    # ------------------------------------------------------------------ #
    # Latency helpers (table-based estimates)
    # ------------------------------------------------------------------ #

    @staticmethod
    def operation_latency(operation: str) -> int:
        """
        Table cost of one *operation* — the key FeaturesAnalyzer counts.

        An operation is the mnemonic, except for a call, where Instruction
        appends the callee ("jal_fixedpt_ln_wdiv32").  Such an operation costs
        the jump plus whatever CALL_BODY_CYCLES records for that callee, which
        is nothing until the callee has been measured.

        An unrecognised operation falls back to INTEGER_LATENCY and is
        reported once.
        """
        op = operation.lower()

        lat = INSTRUCTION_LATENCY.get(op)
        if lat is not None:
            return lat

        # "<call mnemonic>_<callee>" — cost the jump, add the body if known.
        mnemonic, _, callee = op.partition("_")
        if callee and mnemonic in INSTRUCTION_LATENCY:
            return INSTRUCTION_LATENCY[mnemonic] + CALL_BODY_CYCLES.get(callee, 0)

        if op not in _WARNED_UNKNOWN:
            _WARNED_UNKNOWN.add(op)
            print(f"Warning: '{operation}' is not in the CV32E40X cycle table, "
                  f"using default latency {INTEGER_LATENCY}")
        return INTEGER_LATENCY

    @staticmethod
    def instruction_latency(instr: Instruction) -> int:
        """Return the table cycle cost of a single instruction."""
        return Disassembler.operation_latency(instr.operation)

    @staticmethod
    def estimate_block_latency(instructions: list[Instruction]) -> int:
        """
        Naïve sum of per-instruction latencies.

        Stall-free floor: hazards and callee bodies are not included (see the
        table's header comment).
        """
        return sum(Disassembler.instruction_latency(i) for i in instructions)

    @staticmethod
    def estimate_latency_from_counts(counts: "dict[str, float] | FeatureVector") -> float:
        """
        Same estimate expressed as Σ cost(operation) × count(operation).

        Takes either a plain {operation: count} mapping or a FeatureVector,
        whose keys are "<operation>_count" (plus hazard and transition
        features, which have no table cost and are skipped).  This is the
        direct counterpart of RegressionModel.predict: the same features, with
        the manual's cycle counts in place of the learned weights.
        """
        values = counts.values if isinstance(counts, FeatureVector) else counts

        total = 0.0
        for name, count in values.items():
            if not name.endswith("_count") or name.startswith("raw_d"):
                continue
            total += Disassembler.operation_latency(name[: -len("_count")]) * count
        return total