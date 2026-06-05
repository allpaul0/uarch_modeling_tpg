from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .measurement import TeamMeasurement


@dataclass
class Uarch:
    """
    Describes a micro-architecture target (e.g. CV32E40X).
    Can be associated with multiple TeamMeasurements (one per team measured
    on this architecture).
    """
    name: str
    isa: str
    abi: str
    measurements: list["TeamMeasurement"] = field(default_factory=list)

    def add_measurement(self, measurement: "TeamMeasurement") -> None:
        self.measurements.append(measurement)

    def __repr__(self) -> str:
        return (
            f"Uarch(name={self.name!r}, isa={self.isa!r}, abi={self.abi!r}, "
            f"measurements={len(self.measurements)})"
        )