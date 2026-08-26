from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from .isa import ISA

if TYPE_CHECKING:
    from .uarch import Uarch
    from .instruction import Instruction
    from .features import FeatureVector
    from .teamMeasurements import TeamMeasurement


@dataclass
class CompiledTeam:
    """
    A Team compiled for a specific ISA.

    v2 model
    --------
    ``TeamCompiled "1..*" --> "1" ISA : compiled for``
    ``TeamCompiled "1"    *-- "1..*" TeamMeasurement : measured as``
    ``TeamMeasurement "0..*" --> "1" Uarch : on``

    In v1 a CompiledTeam pointed at a Uarch and owned a single measurement.
    Since the compiled code only depends on the ISA, while its timing depends
    on the micro-architecture executing it, one CompiledTeam now holds one
    TeamMeasurement per Uarch it was run on.  The static side of the object
    (``code``, ``instructions``, ``feature_vector``) is therefore shared by
    every uarch implementing the ISA, and is computed only once.

    Attributes
    ----------
    isa:            The ISA this team was compiled for.
    code:           Raw disassembly text of the compiled block.
    instructions:   Ordered list of disassembled Instructions.
    feature_vector: Static feature representation extracted from instructions
                    (ISA-determined, uarch-independent).
    measurements:   uarch name -> TeamMeasurement taken on that uarch.
    """
    isa: ISA
    code: str
    instructions: list["Instruction"] = field(default_factory=list)
    feature_vector: Optional["FeatureVector"] = None
    measurements: dict[str, "TeamMeasurement"] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def isa_name(self) -> str:
        """The ISA string this team was compiled for."""
        return self.isa.name if self.isa is not None else ""

    def add_measurement(
        self,
        measurement: "TeamMeasurement",
        uarch: "Uarch | None" = None,
    ) -> None:
        """
        Attach *measurement*, taken on *uarch*, to this CompiledTeam.

        If *uarch* is omitted the measurement's own ``uarch`` field is used.
        An existing measurement for the same uarch is replaced.

        Raises:
            ValueError: If no uarch can be determined.
        """
        if uarch is None:
            uarch = measurement.uarch
        if uarch is None:
            raise ValueError(
                "A TeamMeasurement must be associated with a Uarch "
                "(pass uarch= or set measurement.uarch)."
            )
        measurement.uarch = uarch
        self.measurements[uarch.name] = measurement

    def get_measurement(self, uarch_name: str) -> "TeamMeasurement | None":
        """Return the measurement taken on *uarch_name*, or None."""
        return self.measurements.get(uarch_name)

    def has_measurement_for(self, uarch_name: str) -> bool:
        return uarch_name in self.measurements

    def uarch_names(self) -> list[str]:
        """Names of every uarch this compiled team has been measured on."""
        return list(self.measurements.keys())

    def uarchs(self) -> list["Uarch"]:
        """Every Uarch this compiled team has been measured on."""
        return [m.uarch for m in self.measurements.values() if m.uarch is not None]

    @property
    def measurement(self) -> "TeamMeasurement | None":
        """
        Legacy single-measurement accessor (v1 API).

        Returns the measurement when exactly one uarch was measured, else
        None — with several uarchs the caller must say which one it means
        via :meth:`get_measurement`.
        """
        if len(self.measurements) == 1:
            return next(iter(self.measurements.values()))
        return None

    # ------------------------------------------------------------------ #
    # Backwards compatibility with v1 pickles
    # ------------------------------------------------------------------ #

    def __setstate__(self, state: dict) -> None:
        # v1 shape: {uarch, code, instructions, feature_vector, measurement}
        if "uarch" in state and "isa" not in state:
            uarch = state.pop("uarch")
            meas = state.pop("measurement", None)
            state["isa"] = uarch.isa if uarch is not None else ISA(name="")
            measurements: dict[str, "TeamMeasurement"] = {}
            if meas is not None and uarch is not None:
                meas.uarch = uarch
                measurements[uarch.name] = meas
            state["measurements"] = measurements
        state.setdefault("measurements", {})
        state.setdefault("feature_vector", None)
        self.__dict__.update(state)

    def __repr__(self) -> str:
        if self.measurements:
            meas_str = ", ".join(
                f"{name}={m.latency:.2f}c"
                for name, m in self.measurements.items()
            )
        else:
            meas_str = "no measurement"
        return (
            f"CompiledTeam(isa={self.isa_name!r}, "
            f"instructions={len(self.instructions)}, "
            f"measurements=[{meas_str}])"
        )


@dataclass
class Team:
    """
    A Team is a basic block node in the TPG, identified by its integer id.

    A Team can be compiled for multiple target ISAs, producing one
    CompiledTeam per ISA.  All ISA-dependent data (code, instructions,
    features) lives on CompiledTeam; all uarch-dependent data (latency)
    lives on the TeamMeasurements owned by that CompiledTeam.

    Attributes
    ----------
    id:             Integer identifier (matches T<id>_start in disassembly).
    compiled_teams: One CompiledTeam per ISA compilation.
    """
    id: int
    compiled_teams: list[CompiledTeam] = field(default_factory=list)

    def add_compiled_team(self, ct: CompiledTeam) -> None:
        self.compiled_teams.append(ct)

    # ------------------------------------------------------------------ #
    # Queries — by ISA (what was compiled)
    # ------------------------------------------------------------------ #

    def get_compiled_for_isa(self, isa: "ISA | str") -> CompiledTeam | None:
        """Return the CompiledTeam compiled for the given ISA, or None."""
        name = isa.name if isinstance(isa, ISA) else isa
        for ct in self.compiled_teams:
            if ct.isa_name == name:
                return ct
        return None

    def isa_names(self) -> list[str]:
        return [ct.isa_name for ct in self.compiled_teams]

    # ------------------------------------------------------------------ #
    # Queries — by uarch (where it ran)
    # ------------------------------------------------------------------ #

    def get_compiled_for_uarch(self, uarch_name: str) -> CompiledTeam | None:
        """
        Return the CompiledTeam that has a measurement on *uarch_name*.

        Kept from the v1 API.  If several ISAs were measured on the same
        uarch, the first one is returned — use
        :meth:`get_compiled_teams_for_uarch` to get them all.
        """
        for ct in self.compiled_teams:
            if ct.has_measurement_for(uarch_name):
                return ct
        return None

    def get_compiled_teams_for_uarch(self, uarch_name: str) -> list[CompiledTeam]:
        """Every CompiledTeam of this Team measured on *uarch_name*."""
        return [ct for ct in self.compiled_teams
                if ct.has_measurement_for(uarch_name)]

    def get_measurement_for_uarch(self, uarch_name: str) -> "TeamMeasurement | None":
        """Return the measurement taken on *uarch_name*, or None."""
        ct = self.get_compiled_for_uarch(uarch_name)
        return ct.get_measurement(uarch_name) if ct is not None else None

    def uarch_names(self) -> list[str]:
        """Every uarch this team has at least one measurement on."""
        names: list[str] = []
        for ct in self.compiled_teams:
            for name in ct.uarch_names():
                if name not in names:
                    names.append(name)
        return names

    def __repr__(self) -> str:
        isas = self.isa_names()
        return (f"Team(id={self.id}, compiled_for={isas}, "
                f"measured_on={self.uarch_names()})")