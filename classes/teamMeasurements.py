from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .uarch import Uarch


@dataclass
class TeamMeasurement:
    """
    Latency statistics for one CompiledTeam executed on one Uarch.

    v2 change
    ---------
    A measurement is now the *hub* of the model: it is the unique point
    identified by a (CompiledTeam, Uarch) pair.  The owning CompiledTeam
    carries the ISA (what was compiled); this object carries the Uarch (where
    it ran).  A CompiledTeam therefore holds one TeamMeasurement per Uarch it
    was executed on.

    ``uarch`` is optional at construction time only so that the object can be
    built before being attached; :meth:`CompiledTeam.add_measurement` fills it
    in.  Prefer passing it explicitly.

    Attributes
    ----------
    latency:         Average cycle count over all runs (AvgCyclesPerTeam).
    nb_measurements: Number of individual runs that produced this average.
    stddev:          Standard deviation of the cycle count across runs.
    uarch:           The micro-architecture this measurement was taken on.
    """
    latency: float
    nb_measurements: int = 0
    stddev: float = 0.0
    uarch: Optional["Uarch"] = None

    @property
    def cv(self) -> float:
        """Coefficient of variation (derived, not stored)."""
        if self.latency == 0.0:
            return 0.0
        return (self.stddev / self.latency) * 100.0

    @property
    def uarch_name(self) -> str:
        """Name of the uarch this measurement belongs to ("" if unattached)."""
        return self.uarch.name if self.uarch is not None else ""

    # ------------------------------------------------------------------ #
    # Backwards compatibility with v1 pickles
    # ------------------------------------------------------------------ #

    def __setstate__(self, state: dict) -> None:
        # v1 measurements had no uarch back-reference; CompiledTeam.__setstate__
        # fills it in from the uarch the v1 CompiledTeam pointed at.
        state.setdefault("uarch", None)
        self.__dict__.update(state)

    def __repr__(self) -> str:
        where = f", uarch={self.uarch_name!r}" if self.uarch is not None else ""
        return (
            f"TeamMeasurement(latency={self.latency:.2f}, "
            f"stddev={self.stddev:.2f}, "
            f"nb_measurements={self.nb_measurements}{where})"
        )