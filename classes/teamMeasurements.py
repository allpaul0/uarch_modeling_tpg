from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TeamMeasurement:
    """
    Latency statistics for one CompiledTeam running on its target Uarch.

    The uarch reference and team identity are held by the owning CompiledTeam,
    so they are not repeated here.

    Attributes
    ----------
    latency:         Average cycle count over all runs (AvgCyclesPerTeam).
    nb_measurements: Number of individual runs that produced this average.
    stddev:          Standard deviation of the cycle count across runs.
    """
    latency: float
    nb_measurements: int = 0
    stddev: float = 0.0

    @property
    def cv(self) -> float:
        """Coefficient of variation (derived, not stored)."""
        if self.latency == 0.0:
            return 0.0
        return (self.stddev / self.latency) * 100.0

    def __repr__(self) -> str:
        return (
            f"TeamMeasurement(latency={self.latency:.2f}, "
            f"stddev={self.stddev:.2f}, "
            f"nb_measurements={self.nb_measurements})"
        )