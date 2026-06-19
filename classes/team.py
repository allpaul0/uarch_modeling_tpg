from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .uarch import Uarch
    from .instruction import Instruction
    from .features import FeatureVector
    from .teamMeasurements import TeamMeasurement


@dataclass
class CompiledTeam:
    """
    A Team compiled for a specific Uarch target.

    One Team produces one CompiledTeam per (uarch, isa, abi) combination.
    The Uarch reference carries the isa and abi used during compilation.

    Attributes
    ----------
    uarch:          The micro-architecture this team was compiled and measured on.
                    Carries the isa and abi used during compilation.
    code:           Raw disassembly text of the compiled block.
    instructions:   Ordered list of disassembled Instructions.
    feature_vector: Feature representation extracted from instructions.
    measurement:    Unique latency measurement taken on the target uarch.
    """
    uarch: "Uarch"
    code: str
    instructions: list["Instruction"] = field(default_factory=list)
    feature_vector: Optional["FeatureVector"] = None
    measurement: Optional["TeamMeasurement"] = None

    def __repr__(self) -> str:
        return (
            f"CompiledTeam(uarch={self.uarch.name!r}, "
            f"instructions={len(self.instructions)}, "
            f"latency={self.measurement.latency:.2f} cycles"
            f" stddev={self.measurement.stddev:.2f}"
            if self.measurement else
            f"CompiledTeam(uarch={self.uarch.name!r}, "
            f"instructions={len(self.instructions)}, no measurement)"
        )


@dataclass
class Team:
    """
    A Team is a basic block node in the TPG, identified by its integer id.

    A Team can be compiled for multiple target uarchs, producing one
    CompiledTeam per target.  All ISA-dependent data (code, instructions,
    features, latency) lives on CompiledTeam, not here.

    Attributes
    ----------
    id:             Integer identifier (matches T<id>_start in disassembly).
    compiled_teams: One CompiledTeam per (uarch, isa, abi) compilation.
    """
    id: int
    compiled_teams: list[CompiledTeam] = field(default_factory=list)

    def add_compiled_team(self, ct: CompiledTeam) -> None:
        self.compiled_teams.append(ct)

    def get_compiled_for_uarch(self, uarch_name: str) -> CompiledTeam | None:
        """Return the CompiledTeam targeting the given uarch, or None."""
        for ct in self.compiled_teams:
            if ct.uarch.name == uarch_name:
                return ct
        return None

    def __repr__(self) -> str:
        uarchs = [ct.uarch.name for ct in self.compiled_teams]
        return f"Team(id={self.id}, compiled_for={uarchs})"