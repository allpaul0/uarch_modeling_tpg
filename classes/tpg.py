from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .team import Team

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