from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .team import Team


@dataclass
class ClassLatencyPair:
    """
    Whole-TPG latency for one class / graph traversal under the two
    instrumentation modes measured in ``latencies.json``.

    Attributes
    ----------
    class_id:   Traversal id (matches the ``[k] ->`` mapping in LE_states.h
                and the "Class" field in the JSON).
    tpg_only:   AvgCyclesPerClass from ``instrTPG`` (only the TPG instrumented).
    tpg_teams:  AvgCyclesPerClass from ``instrTeams_instrTPG`` (TPG *and* teams
                instrumented — carries the per-team probe overcost).
    tpg_only_stddev / tpg_teams_stddev: matching StddevCyclesPerClass values.
    """
    class_id: int
    tpg_only: float
    tpg_teams: float
    tpg_only_stddev: float = 0.0
    tpg_teams_stddev: float = 0.0

    @property
    def overcost(self) -> float:
        """Absolute instrumentation overcost (cycles) of measuring teams."""
        return self.tpg_teams - self.tpg_only

    @property
    def overcost_pct(self) -> float:
        """Overcost as a percentage of the TPG-only latency."""
        return (self.overcost / self.tpg_only * 100.0) if self.tpg_only else 0.0


@dataclass
class TPG:
    """
    Represents a Tangled Program Graph — the top-level container
    that holds one or more Teams.

    Attributes
    ----------
    source_path : str
        Absolute path of the seed directory on disk.  Used as the unique
        identity key in Database.tpgs and in loaded_keys — guaranteed to
        be unambiguous even when loading from multiple roots or when two
        seed directories share the same basename.
    name : str
        Human-readable display name (seed-directory basename).  Used only
        for printing; never used as a dict key or dedup guard.
    dtype : str
        Data type tag (e.g. "fixedpt").
    """
    source_path: str          # unique key — absolute path of seed dir
    name: str                 # display only — seed-dir basename
    dtype: str
    gpis: list[str] = field(default_factory=list)
    teams: list["Team"] = field(default_factory=list)

    # Graph traversals parsed from outLogs/precalcul/LE_states.h.
    # Maps traversal/class id -> ordered list of team ids visited.
    # ISA/uarch-independent (it describes graph structure), so it lives on
    # the TPG rather than per-compilation.
    traversals: dict[int, list[int]] = field(default_factory=dict)

    # Per-uarch whole-TPG class latencies (both instrumentation modes).
    # Keyed by uarch name, then by class/traversal id.  Populated at load
    # time from each uarch's latencies.json.
    class_latencies: dict[str, dict[int, "ClassLatencyPair"]] = field(
        default_factory=dict
    )

    def add_team(self, team: "Team") -> None:
        self.teams.append(team)

    def get_team(self, team_id: int) -> "Team | None":
        for team in self.teams:
            if team.id == team_id:
                return team
        return None

    def __repr__(self) -> str:
        return (
            f"TPG(name={self.name!r}, source_path={self.source_path!r}, "
            f"dtype={self.dtype!r}, teams=[{len(self.teams)} team(s)])"
        )