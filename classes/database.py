"""
classes/database.py — central Database that owns all Uarchs and TPGs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tpg import TPG
    from .team import Team, CompiledTeam
    from .uarch import Uarch


@dataclass
class Database:
    """
    Top-level container for the entire multi-TPG, multi-Uarch dataset.

    Structure
    ---------
    uarchs : dict[str, Uarch]
        Registry of known Uarch objects, keyed by uarch name.  Ensures all
        CompiledTeams targeting the same uarch share one Uarch instance.

    tpgs : dict[str, TPG]
        Keyed by TPG.source_path (absolute path of the seed directory).
        Using the full path — not the basename — guarantees uniqueness even
        when loading from multiple roots or when seeds share a basename.

    loaded_keys : set[str]
        Deduplication guard.  Tracks "<source_path>|<uarch_name>" pairs
        that have already been ingested so re-loading is a no-op.
    """

    uarchs:      dict[str, "Uarch"] = field(default_factory=dict)
    tpgs:        dict[str, "TPG"]   = field(default_factory=dict)  # key = source_path
    loaded_keys: set[str]           = field(default_factory=set)

    # ------------------------------------------------------------------ #
    # Insertion helpers
    # ------------------------------------------------------------------ #

    def get_or_create_uarch(self, name: str, isa: str, abi: str) -> "Uarch":
        from .uarch import Uarch
        if name not in self.uarchs:
            self.uarchs[name] = Uarch(name=name, isa=isa, abi=abi)
        return self.uarchs[name]

    def get_or_create_tpg(
        self,
        source_path: str,   # unique key — absolute path of seed dir
        name: str,          # display name — seed-dir basename
        dtype: str,
    ) -> "TPG":
        from .tpg import TPG
        if source_path not in self.tpgs:
            self.tpgs[source_path] = TPG(
                source_path=source_path,
                name=name,
                dtype=dtype,
            )
        return self.tpgs[source_path]

    def is_loaded(self, source_path: str, uarch_name: str) -> bool:
        return f"{source_path}|{uarch_name}" in self.loaded_keys

    def mark_loaded(self, source_path: str, uarch_name: str) -> None:
        self.loaded_keys.add(f"{source_path}|{uarch_name}")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_compiled_teams_for_uarch(
        self, uarch_name: str
    ) -> list[tuple["TPG", int, "CompiledTeam"]]:
        """
        Return all (tpg, team_id, CompiledTeam) triples across all TPGs for
        the given uarch.

        The TPG is included so callers can unambiguously identify which TPG
        each team belongs to (team IDs are only unique within a TPG).
        """
        result: list[tuple["TPG", int, "CompiledTeam"]] = []
        for tpg in self.tpgs.values():
            for team in tpg.teams:
                ct = team.get_compiled_for_uarch(uarch_name)
                if ct is not None:
                    result.append((tpg, team.id, ct))
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Database] Saved to {path}  "
              f"({len(self.tpgs)} TPGs, {len(self.uarchs)} uarchs)")

    @staticmethod
    def load(path: str | Path) -> "Database":
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

    def _count_compiled_teams_for_uarch(self, uarch_name: str) -> int:
        return sum(
            1
            for tpg in self.tpgs.values()
            for team in tpg.teams
            for ct in team.compiled_teams
            if ct.uarch.name == uarch_name
        )

    def print_summary(self) -> None:
        total_teams = sum(len(tpg.teams) for tpg in self.tpgs.values())

        print("\n" + "═" * 70)
        print(f"  DATABASE SUMMARY  —  {len(self.tpgs)} TPGs  ·  {len(self.uarchs)} Uarchs")
        print(f"  ·  Teams (unique): {total_teams}")
        print("═" * 70)

        print("\n  Uarchs:")
        for uarch in self.uarchs.values():
            n = self._count_compiled_teams_for_uarch(uarch.name)
            print(f"    {uarch.name}  isa={uarch.isa}  abi={uarch.abi}"
                  f"  ({n} compiled teams)")

        print(f"\n  TPGs ({len(self.tpgs)}):")
        for tpg in self.tpgs.values():
            uarch_names = sorted({
                ct.uarch.name
                for team in tpg.teams
                for ct in team.compiled_teams
            })
            # Display the human-readable name; source_path shown for traceability
            print(f"    {tpg.name}")
            print(f"      path={tpg.source_path}")
            print(f"      dtype={tpg.dtype}"
                  f"  teams={len(tpg.teams)}"
                  f"  uarchs={uarch_names}")
        print("═" * 70)

    def print_uarch(self, uarch_name: str, max_teams: int | None = None) -> None:
        """
        Print every CompiledTeam targeting uarch_name: code, instructions,
        latency, and feature vector (if already computed).
        """
        uarch = self.uarchs.get(uarch_name)
        if uarch is None:
            print(f"[Database] Unknown uarch: {uarch_name!r}")
            return

        triples = self.get_compiled_teams_for_uarch(uarch_name)
        if max_teams is not None:
            triples = triples[:max_teams]

        bar = "─" * 70
        print(f"\n{'═'*70}")
        print(f"  UARCH: {uarch.name}  |  ISA: {uarch.isa}  |  ABI: {uarch.abi}")
        print(f"  Showing {len(triples)} compiled team(s)")
        print(f"{'═'*70}")

        for tpg, team_id, ct in triples:
            # tpg is the direct owner — no search needed, no misidentification
            meas = ct.measurement
            print(f"\n{bar}")
            print(f"  Team {team_id}  |  TPG: {tpg.name}  |  path: {tpg.source_path}")
            if meas:
                print(f"  Latency : {meas.latency:.2f} cycles"
                      f"  stddev={meas.stddev:.2f}"
                      f"  nb_measurements={meas.nb_measurements}")
            print(f"  asm:")
            print(ct.code)
            print(f"\n  Instructions ({len(ct.instructions)}):")
            for instr in ct.instructions:
                print(f"    {instr.mnemonic:<12} {' '.join(instr.operands)}")

            if ct.feature_vector and ct.feature_vector.values:
                fv = ct.feature_vector
                non_zero = sorted(
                    ((k, v) for k, v in fv.values.items() if v != 0.0),
                    key=lambda kv: -kv[1],
                )
                print(f"  FeatureVector ({len(non_zero)} non-zero features):")
                for k, v in non_zero[:15]:
                    print(f"    {k:<35} {v:.1f}")
                if len(non_zero) > 15:
                    print(f"    … ({len(non_zero) - 15} more)")
            else:
                print(f"  FeatureVector: not yet computed")
        print(bar)