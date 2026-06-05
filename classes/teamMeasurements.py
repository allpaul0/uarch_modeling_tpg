from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .uarch import Uarch


@dataclass
class TeamMeasurement:
    """
    A single measured latency for a Team running on a given Uarch.
    """
    uarch: "Uarch"
    id_team: int
    latency: float

    def __repr__(self) -> str:
        return (
            f"uarch={self.uarch.name!r}, "
            f"TeamMeasurement(id_team={self.id_team}), "
            f"latency={self.latency})"
        )