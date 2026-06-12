from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .measurement import TeamMeasurement
    from .features import FeatureVector


@dataclass
class Team:
    """
    A Team corresponds to one basic block in the TPG.

    Attributes
    ----------
    id:                   Integer identifier (matches T<id>_start in the
                          disassembly).
    code:                 Raw disassembly text of the block, exactly as it
                          appears between the T<id>_start and T<id>_end
                          label lines in the objdump output.
    nb_team_measurements: Total number of individual measurement runs across
                          all TeamMeasurements attached to this team.  This
                          is the sum of TeamMeasurement.nb_measurements, not
                          simply the number of TeamMeasurement objects.
    instructions:         Ordered list of Instructions in the block.
    measurements:         List of TeamMeasurements (one per uarch typically).
    feature_vector:       Extracted feature representation of the block.
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
        # nb_team_measurements is the total run count, not the list length
        self.nb_team_measurements = sum(
            m.nb_measurements for m in self.measurements
        )

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
        # Show only the first line of code so repr stays readable in a loop
        first_line = self.code.splitlines()[0] if self.code else ""
        return (
            f"Team(id={self.id}, "
            f"instructions={len(self.instructions)}, "
            f"nb_measurements={self.nb_team_measurements}, "
            f"code_start={first_line!r})"
        )