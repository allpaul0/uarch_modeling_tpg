from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .uarch import Uarch


@dataclass
class TeamMeasurement:
    """
    A single measured latency for a Team running on a given Uarch.

    Attributes
    ----------
    uarch:           Micro-architecture the measurement was taken on.
    id_team:         Team identifier.
    latency:         Average cycle count over all runs (AvgCyclesPerTeam).
    nb_measurements: Number of individual runs that produced this average
                     (Count in the JSON results file).
    stddev:          Standard deviation of the cycle count across runs
                     (StddevCyclesPerTeam).
    cv:              Coefficient of variation (StddevCyclesPerTeam /
                     AvgCyclesPerTeam × 100), expressed as a percentage.
    """
    uarch: "Uarch"
    id_team: int
    latency: float
    nb_measurements: int = 0
    stddev: float = 0.0
    cv: float = 0.0

    def __repr__(self) -> str:
        return (
            f"TeamMeasurement(id_team={self.id_team}, "
            f"uarch={self.uarch.name!r}, "
            f"latency={self.latency:.2f}, "
            f"nb_measurements={self.nb_measurements}, "
            f"stddev={self.stddev:.2f}, cv={self.cv:.4f})"
        )