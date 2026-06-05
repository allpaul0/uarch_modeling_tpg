from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .measurement import TeamMeasurement
    from .features import FeatureVector


@dataclass
class Instruction:
    """
    A single disassembled RISC-V instruction belonging to a Team.
    """
    mnemonic: str
    operands: list[str]
    raw: str = ""

    def __repr__(self) -> str:
        return (
            f"Instruction(mnemonic={self.mnemonic!r}, "
            f"operands={self.operands})"
        )

    @staticmethod
    def parse(asm: str) -> "Instruction":
        """
        Parse a single RISC-V assembly string such as:
            'sw  a5,-172(s0)'
            'addi a5,a5,4'
            'li   a5,0'
        """
        asm = asm.strip()
        parts = asm.split(None, 1)
        mnemonic = parts[0]
        operands_part = parts[1] if len(parts) > 1 else ""

        raw_operands = [op.strip() for op in operands_part.split(",") if op.strip()]

        operands: list[str] = []

        for op in raw_operands:
            # Case 1: pure immediate (e.g. 4, -172, 0)
            if re.fullmatch(r"-?\d+", op):
                operands.append("CONST")
                continue

            # Case 2: memory operand like -172(s0)
            m = re.fullmatch(r"(-?\d+)\((\w+)\)", op)
            if m:
                # split into CONST + register
                operands.append("CONST")
                operands.append(m.group(2))
                continue

            # Case 3: normal register (a5, s0, etc.)
            operands.append(op)

        return Instruction(
            mnemonic=mnemonic,
            operands=operands,
            raw=asm,
        )


@dataclass
class Team:
    """
    A Team corresponds to one code block / program group in the TPG.
    It owns a list of Instructions, a list of TeamMeasurements,
    and one FeatureVector.
    """
    id: int
    code: str
    nb_team_measurements: int = 0
    instructions: list[Instruction] = field(default_factory=list)
    measurements: list["TeamMeasurement"] = field(default_factory=list)
    feature_vector: Optional["FeatureVector"] = None

    # ------------------------------------------------------------------ #
    # Instruction helpers
    # ------------------------------------------------------------------ #

    def add_instruction(self, instr: Instruction) -> None:
        self.instructions.append(instr)

    def set_instructions(self, instrs: list[Instruction]) -> None:
        self.instructions = instrs

    # ------------------------------------------------------------------ #
    # Measurement helpers
    # ------------------------------------------------------------------ #

    def add_measurement(self, measurement: "TeamMeasurement") -> None:
        self.measurements.append(measurement)
        self.nb_team_measurements = len(self.measurements)

    def get_latencies(self) -> list[float]:
        """Return all recorded latencies for this team."""
        return [m.latency for m in self.measurements]

    def mean_latency(self) -> Optional[float]:
        lats = self.get_latencies()
        return sum(lats) / len(lats) if lats else None

    # ------------------------------------------------------------------ #
    # Feature vector
    # ------------------------------------------------------------------ #

    def set_feature_vector(self, fv: "FeatureVector") -> None:
        self.feature_vector = fv

    def __repr__(self) -> str:
        return (
            f"Team(id={self.id}, code={self.code!r}, "
            f"instructions={len(self.instructions)}, "
            f"measurements={self.nb_team_measurements})"
        )