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
#      Captured by  (?:\s+<\S+>)?   before the optional # comment.
#
#   2. Hash comment: "lw a7,4(a3) # 20000004 <_sp+…>"
#      Captured by  (?:\s*#.*)?
#
# The non-greedy (.+?) stops before either optional suffix, so group 1
# always contains only mnemonic + clean operands.
_LINE_RE = re.compile(
    r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(.+?)(?:\s+<\S+>)?(?:\s*#.*)?$"
)

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
class TPGLatencyData:
    """
    All latency information parsed from one JSON results file.

    Attributes
    ----------
    simulator:      Simulator / uarch name  (e.g. "cv32e40x_im2_zba_zbb").
    isa:            ISA string              (e.g. "rv32ic_zicsr_zmmul_zba_zbb").
    abi:            ABI string              (e.g. "ilp32").
    dtype:          Data type               (e.g. "fixedpt").
    tpg_mean_lat:   Mean latency of the whole TPG (cycles).
    tpg_stddev_lat: Stddev of the whole-TPG latency (cycles).
    team_latencies: Dict mapping team_id → TeamLatency.  Only teams that
                    appear in the JSON "Teams" array are present; zero-
                    latency teams are absent.
    """
    simulator: str
    isa: str
    abi: str
    dtype: str
    tpg_mean_lat: float
    tpg_stddev_lat: float
    team_latencies: dict[int, TeamLatency] = field(default_factory=dict)

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
                instructions.append(Instruction.parse(m.group(1).strip()))
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
                    instructions.append(Instruction.parse(instr_m.group(1).strip()))

        return blocks

    # ------------------------------------------------------------------ #
    # Public API — JSON latency parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_latency_json(path: str) -> TPGLatencyData:
        """
        Parse a JSON results file.  Only ``instrTeams_instrTPG.Teams`` is used
        for per-team latencies.
        """
        with open(path, "r") as fh:
            data: dict = json.load(fh)

        section = data["instrTeams_instrTPG"]
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

        return TPGLatencyData(
            simulator=data["simulator"],
            isa=data["isa"],
            abi=data["abi"],
            dtype=data["dtype"],
            tpg_mean_lat=float(section["tpg_mean_lat"]),
            tpg_stddev_lat=float(section["tpg_stddev_lat"]),
            team_latencies=team_latencies,
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