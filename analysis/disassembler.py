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
#   <addr>:  <machine-code>   <mnemonic> [operands]   [# optional comment]
_LINE_RE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(.+?)(?:\s*#.*)?$")

# Matches an exact team label line, e.g. "000007c2 <T0_start>:"
# The leading ^ and trailing \s*$ prevent false positives from inline
# references like "# 1264 <T21_start>" that appear inside instruction lines.
_TEAM_LABEL_RE = re.compile(r"^[0-9a-f]{8} <T(\d+)_(start|end)>:\s*$")

# The first instruction inside T_start is always "csrr <rd>,mcycle" — a
# timing probe inserted by the test harness.  It must be stripped from the
# instruction list and from Team.code.
_MCYCLE_PROBE_RE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+csrr\s+\w+,mcycle")


@dataclass
class TeamBlock:
    """
    Everything extracted for one team from the disassembly file.

    Attributes
    ----------
    team_id:      Integer from the T<N>_start label.
    instructions: Ordered list of Instructions (timing probe excluded).
    code:         Raw disassembly text of the block — the label lines plus
                  every instruction line, exactly as printed by objdump,
                  with the ``csrr mcycle`` timing probe line stripped.
                  This is the value stored in Team.code.
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
                    appear in the JSON "Teams" array (non-zero latency) are
                    present; zero-latency teams are absent.
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
    Parses raw RISC-V assembly text (e.g. output of objdump -d) into
    Instruction objects.

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

        Args:
            code: Raw text from objdump (or similar disassembler).

        Returns:
            Ordered list of Instruction objects.
        """
        instructions: list[Instruction] = []
        for line in code.splitlines():
            m = _LINE_RE.search(line)
            if m:
                instructions.append(Instruction.parse(m.group(1).strip()))
        return instructions

    @staticmethod
    def parse_file(path: str) -> list[Instruction]:
        """
        Convenience wrapper: read a file and call parse_assembly.

        Args:
            path: Path to the objdump output file.

        Returns:
            Ordered list of Instruction objects.
        """
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

        For each team the method returns a :class:`TeamBlock` containing:

        * ``instructions`` — parsed :class:`Instruction` objects, with the
          ``csrr mcycle`` timing probe removed.
        * ``code`` — the raw objdump text of the block (label lines + all
          instruction lines), again with the probe line removed.  This is
          what callers store in ``Team.code`` so the original disassembly is
          always accessible.

        Lines that fall between a ``T_end`` and the next ``T_start`` (harness
        glue code) are silently ignored.

        Args:
            path: Path to the objdump output file.

        Returns:
            ``{team_id: TeamBlock}`` — one entry per team found in the file.
        """
        blocks: dict[int, TeamBlock] = {}

        current_id: int | None = None
        probe_consumed: bool = False
        raw_lines: list[str] = []          # accumulates text for Team.code
        instructions: list[Instruction] = []

        with open(path, "r") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                label_m = _TEAM_LABEL_RE.match(line)
                if label_m:
                    tid    = int(label_m.group(1))
                    marker = label_m.group(2)  # "start" or "end"

                    if marker == "start":
                        current_id     = tid
                        probe_consumed = False
                        raw_lines      = [line]   # include the label line
                        instructions   = []

                    elif marker == "end" and current_id is not None:
                        raw_lines.append(line)    # include the end label line
                        blocks[current_id] = TeamBlock(
                            team_id=current_id,
                            instructions=instructions,
                            code="\n".join(raw_lines),
                        )
                        current_id = None
                    continue

                if current_id is None:
                    continue   # harness glue between teams — skip

                # First real line after T_start: the mcycle timing probe
                if not probe_consumed:
                    if _MCYCLE_PROBE_RE.match(line):
                        probe_consumed = True   # strip probe from code & instrs
                    # skip blank lines and the probe line itself
                    continue

                # Normal instruction line inside the block
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
        Parse a JSON results file produced by the CV32E40X inference pipeline.

        Only the ``instrTeams_instrTPG.Teams`` array is used for per-team
        latencies.  Teams absent from that array have zero / unmeasured
        latency and will not appear in ``TPGLatencyData.team_latencies``.

        Args:
            path: Path to the JSON results file.

        Returns:
            A populated :class:`TPGLatencyData` instance.
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
    # Latency helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def instruction_latency(instr: Instruction) -> int:
        """Return the estimated cycle latency for a single instruction."""
        lat = INSTRUCTION_LATENCY.get(instr.mnemonic.lower())
        if lat is None:
            print(
                f"Warning: unknown mnemonic '{instr.mnemonic}', "
                f"using default latency {INTEGER_LATENCY}"
            )
            return INTEGER_LATENCY
        return lat

    @staticmethod
    def estimate_block_latency(instructions: list[Instruction]) -> int:
        """
        Naïve basic-block latency estimate: sum of per-instruction latencies.

        Args:
            instructions: Ordered list of Instructions in the block.

        Returns:
            Total estimated cycles.
        """
        return sum(Disassembler.instruction_latency(i) for i in instructions)