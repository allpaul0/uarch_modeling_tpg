from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .team import Team

@dataclass
class TPG:
    """
    Represents a Tangeled Program Graph — the top-level container
    that holds one or more Teams.
    """
    name: str
    dtype: str
    gpis: list[str] = field(default_factory=list)
    teams: list[Team] = field(default_factory=list)

    def add_team(self, team: "Team") -> None:
        self.teams.append(team)

    def get_team(self, team_id: int) -> "Team | None":
        for team in self.teams:
            if team.id == team_id:
                return team
        return None

    def __repr__(self) -> str:
        return (
            f"TPG(name={self.name!r}, dtype={self.dtype!r}, "
            f"gpis={self.gpis}, teams=[{len(self.teams)} team(s)])"
        )