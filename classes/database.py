"""
classes/database.py — central Database that owns all Uarchs and TPGs.

The Database is the single source of truth.  It is:
  - serialisable / deserialisable via pickle  (preserves the full object graph)
  - queryable by uarch or by (tpg, team)
  - deduplication-aware: loading the same TPG/uarch data twice is a no-op
"""

from __future__ import annotations

import pickle
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tpg import TPG
    from .team import Team
    from .uarch import Uarch
    from .teamMeasurements import TeamMeasurement


@dataclass
class Database:
    """
    Top-level container for the entire multi-TPG, multi-Uarch dataset.

    Structure
    ---------
    uarchs : dict[str, Uarch]
        Keyed by ``Uarch.name`` (= simulator field from the JSON, e.g.
        ``"cv32e40x_im2_zba_zbb"``).  One Uarch object per distinct
        micro-architecture; it aggregates *all* TeamMeasurements taken on it.

    tpgs : dict[str, TPG]
        Keyed by TPG name (= seed-directory basename).  Each TPG owns its
        Teams.  Teams hold their Instructions, FeatureVector, and the list of
        TeamMeasurements that link them back to specific Uarchs.

    loaded_keys : set[str]
        Tracks ``"<tpg_name>|<uarch_name>"`` strings for every (TPG, Uarch)
        pair that has already been ingested.  Used by the deduplication check
        so that re-loading the same results folder is a no-op.
    """

    uarchs: dict[str, "Uarch"] = field(default_factory=dict)
    tpgs:   dict[str, "TPG"]   = field(default_factory=dict)
    loaded_keys: set[str]      = field(default_factory=set)

    # ------------------------------------------------------------------ #
    # Insertion helpers
    # ------------------------------------------------------------------ #

    def get_or_create_uarch(self, name: str, isa: str, abi: str) -> "Uarch":
        """Return the existing Uarch or create and register a new one."""
        from .uarch import Uarch
        if name not in self.uarchs:
            self.uarchs[name] = Uarch(name=name, isa=isa, abi=abi)
        return self.uarchs[name]

    def get_or_create_tpg(self, name: str, dtype: str) -> "TPG":
        """Return the existing TPG or create and register a new one."""
        from .tpg import TPG
        if name not in self.tpgs:
            self.tpgs[name] = TPG(name=name, dtype=dtype)
        return self.tpgs[name]

    def is_loaded(self, tpg_name: str, uarch_name: str) -> bool:
        """Return True if this (tpg, uarch) pair is already in the database."""
        return f"{tpg_name}|{uarch_name}" in self.loaded_keys

    def mark_loaded(self, tpg_name: str, uarch_name: str) -> None:
        """Record that this (tpg, uarch) pair has been ingested."""
        self.loaded_keys.add(f"{tpg_name}|{uarch_name}")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_teams_for_uarch(self, uarch_name: str) -> list["Team"]:
        """
        Return all Teams across all TPGs that have at least one measurement
        on the given Uarch.
        """
        result: list["Team"] = []
        for tpg in self.tpgs.values():
            for team in tpg.teams:
                if any(m.uarch.name == uarch_name for m in team.measurements):
                    result.append(team)
        return result

    def get_measurement(
        self, tpg_name: str, team_id: int, uarch_name: str
    ) -> "TeamMeasurement | None":
        """Return the specific measurement for a (tpg, team, uarch) triple."""
        tpg = self.tpgs.get(tpg_name)
        if tpg is None:
            return None
        team = tpg.get_team(team_id)
        if team is None:
            return None
        for m in team.measurements:
            if m.uarch.name == uarch_name:
                return m
        return None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Serialise the entire Database to a pickle file.

        The full object graph (cross-references between Teams,
        TeamMeasurements, and Uarchs) is preserved by pickle.

        Args:
            path: Destination file path (conventionally ``*.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Database] Saved to {path}  "
              f"({len(self.tpgs)} TPGs, {len(self.uarchs)} uarchs)")

    @staticmethod
    def load(path: str | Path) -> "Database":
        """
        Deserialise a Database from a pickle file.

        Args:
            path: Path to a file previously written by :meth:`save`.

        Returns:
            The restored Database instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found: {path}")
        with open(path, "rb") as fh:
            db: Database = pickle.load(fh)
        print(f"[Database] Loaded from {path}  "
              f"({len(db.tpgs)} TPGs, {len(db.uarchs)} uarchs)")
        return db

    # ------------------------------------------------------------------ #
    # Pretty-printing
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print a high-level overview: uarchs, TPGs, and team counts."""
        print("\n" + "═" * 70)
        print(f"  DATABASE SUMMARY  —  {len(self.tpgs)} TPGs  ·  {len(self.uarchs)} Uarchs")
        print("═" * 70)

        print("\n  Uarchs:")
        for uarch in self.uarchs.values():
            print(f"    {uarch.name}  isa={uarch.isa}  abi={uarch.abi}"
                  f"  ({len(uarch.measurements)} measurements)")

        print(f"\n  TPGs ({len(self.tpgs)}):")
        for tpg in self.tpgs.values():
            uarch_names = sorted({
                m.uarch.name
                for team in tpg.teams
                for m in team.measurements
            })
            print(f"    {tpg.name}")
            print(f"      dtype={tpg.dtype}  teams={len(tpg.teams)}"
                  f"  uarchs={uarch_names}")
        print("═" * 70)

    def print_uarch(self, uarch_name: str, max_teams: int | None = None) -> None:
        """
        Print every Team that has a measurement on *uarch_name*, showing its
        instructions, latency, and feature vector.

        Args:
            uarch_name: Key in ``self.uarchs`` (simulator name).
            max_teams:  Cap the number of teams printed.  None = all.
        """
        uarch = self.uarchs.get(uarch_name)
        if uarch is None:
            print(f"[Database] Unknown uarch: {uarch_name!r}")
            return

        teams = self.get_teams_for_uarch(uarch_name)
        if max_teams is not None:
            teams = teams[:max_teams]

        bar = "─" * 70
        print(f"\n{'═'*70}")
        print(f"  UARCH: {uarch.name}  |  ISA: {uarch.isa}  |  ABI: {uarch.abi}")
        print(f"  Showing {len(teams)} team(s) with measurements on this uarch")
        print(f"{'═'*70}")

        for team in teams:
            meas = next(m for m in team.measurements if m.uarch.name == uarch_name)
            # Find which TPG owns this team
            tpg_name = next(
                (tpg.name for tpg in self.tpgs.values()
                 if any(t.id == team.id for t in tpg.teams)),
                "?"
            )
            print(f"\n{bar}")
            print(f"  Team {team.id}  |  TPG: {tpg_name}")
            print(f"  Latency : {meas.latency:.2f} cycles"
                  f"  stddev={meas.stddev:.2f}"
                  f"  nb_measurements={meas.nb_measurements}")
            print(f"  Instructions ({len(team.instructions)}):")
            for instr in team.instructions:
                print(f"    {instr.mnemonic:<12} {' '.join(instr.operands)}")
            if team.feature_vector and team.feature_vector.values:
                fv = team.feature_vector
                # Show only non-zero features, sorted by value desc
                non_zero = sorted(
                    ((k, v) for k, v in fv.values.items() if v != 0.0),
                    key=lambda kv: -kv[1],
                )
                print(f"  FeatureVector ({len(non_zero)} non-zero features):")
                for k, v in non_zero[:15]:
                    print(f"    {k:<35} {v:.1f}")
                if len(non_zero) > 15:
                    print(f"    … ({len(non_zero) - 15} more)")
        print(bar)