from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .uarch import Uarch


@dataclass
class DispatchMeasurement:
    """
    Latency of the TPG *dispatch* for one dispatch size, on one Uarch.

    A dispatch is the control step that runs after a Team's programs have been
    evaluated: the program results are compared, the highest one wins, and
    control moves to the winner's destination (another Team, or an Action).
    Its cost is driven by how many programs have to be compared — the
    ``DispatchSize`` — not by any instruction stream that can be attributed to
    a Team, which is why it is modelled separately from CompiledTeams.

    The instrumented build reports one aggregate per distinct dispatch size
    per TPG (``instrDispatch_instrTeams_instrTPG.Dispatches``), so this object
    is an average over ``nb_measurements`` dispatch events of that size.

    Attributes
    ----------
    dispatch_size:   Number of programs compared in the dispatch.
    latency:         Average cycle count (AvgCyclesPerDispatch).
    nb_measurements: Number of dispatch events averaged (Count).
    stddev:          StddevCyclesPerDispatch across those events.
    uarch:           Micro-architecture the measurement was taken on.
    """
    dispatch_size: int
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
    def sem(self) -> float:
        """
        Standard error of this mean — the precision of the label.

        ``stddev / sqrt(nb_measurements)``: how tightly the reported average
        pins down the true mean cost of a dispatch of this size.
        """
        if self.nb_measurements <= 0:
            return 0.0
        return self.stddev / (self.nb_measurements ** 0.5)

    @property
    def uarch_name(self) -> str:
        return self.uarch.name if self.uarch is not None else ""

    def __setstate__(self, state: dict) -> None:
        state.setdefault("uarch", None)
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"DispatchMeasurement(size={self.dispatch_size}, "
            f"latency={self.latency:.2f}, stddev={self.stddev:.2f}, "
            f"nb_measurements={self.nb_measurements})"
        )