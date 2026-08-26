from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

from classes.instruction import Instruction


# ---------------------------------------------------------------------------
# Default / table-based latencies (CV32E40X)
# ---------------------------------------------------------------------------

INTEGER_LATENCY = 1  # default for any RV32I integer instruction

INSTRUCTION_LATENCY: dict[str, int] = {
    # Memory
    "lw": 1, "lh": 1, "lb": 1, "lhu": 1, "lbu": 1,
    "sw": 1, "sh": 1, "sb": 1,
    # Control flow
    "jal": 2, "jalr": 2,
    # Branches
    "beq": 1, "bne": 1, "blt": 1, "bge": 1, "bltu": 1, "bgeu": 1,
    # Multiply
    "mul": 1, "mulh": 4, "mulhu": 4, "mulhsu": 4,
    # Division
    "div": 35, "divu": 35, "rem": 35, "remu": 35,
    # CSR
    "csrrw": 1, "csrrs": 1, "csrrc": 1,
    "csrrwi": 1, "csrrsi": 1, "csrrci": 1,
    # System / fence
    "mret": 2, "fence": 5, "fence.i": 5,
    # Bit-manip (Zba/Zbb/Zbc/Zbs)
    "clz": 1, "ctz": 1, "pcnt": 1, "max": 1, "min": 1,
}

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

        Returns:
            ``{team_id: TeamBlock}`` — one entry per team in the file.
        """
        blocks: dict[int, TeamBlock] = {}
        current_id: int | None = None
        probe_consumed: bool = False
        raw_lines: list[str] = []
        instructions: list[Instruction] = []

        with open(path, "r") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                label_m = _TEAM_LABEL_RE.match(line)
                if label_m:
                    tid    = int(label_m.group(1))
                    marker = label_m.group(2)

                    if marker == "start":
                        current_id     = tid
                        probe_consumed = False
                        raw_lines      = [line]
                        instructions   = []
                    elif marker == "end" and current_id is not None:
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
    def instruction_latency(instr: Instruction) -> int:
        """Return the estimated cycle latency for a single instruction."""
        lat = INSTRUCTION_LATENCY.get(instr.mnemonic.lower())
        if lat is None:
            print(f"Warning: unknown mnemonic '{instr.mnemonic}', "
                  f"using default latency {INTEGER_LATENCY}")
            return INTEGER_LATENCY
        return lat

    @staticmethod
    def estimate_block_latency(instructions: list[Instruction]) -> int:
        """Naïve sum of per-instruction latencies."""
        return sum(Disassembler.instruction_latency(i) for i in instructions)