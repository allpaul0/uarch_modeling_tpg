"""
classes/database.py — central Database that owns all ISAs, Uarchs and TPGs.
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
    from .isa import ISA
    from .teamMeasurements import TeamMeasurement


@dataclass
class Database:
    """
    Top-level container for the entire multi-TPG, multi-ISA, multi-Uarch
    dataset.

    Structure
    ---------
    isas : dict[str, ISA]
        Registry of known ISA objects, keyed by ISA name.  Ensures all
        CompiledTeams compiled for the same ISA — and all Uarchs
        implementing it — share one ISA instance.

    uarchs : dict[str, Uarch]
        Registry of known Uarch objects, keyed by uarch name.  Ensures all
        TeamMeasurements taken on the same uarch share one Uarch instance.

    tpgs : dict[str, TPG]
        Keyed by TPG.source_path (absolute path of the seed directory).
        Using the full path — not the basename — guarantees uniqueness even
        when loading from multiple roots or when seeds share a basename.

    loaded_keys : set[str]
        Deduplication guard.  Tracks "<source_path>|<uarch_name>|<isa_name>"
        triples that have already been ingested so re-loading is a no-op.
        v1 keys ("<source_path>|<uarch_name>") are still honoured.
    """

    uarchs:      dict[str, "Uarch"] = field(default_factory=dict)
    tpgs:        dict[str, "TPG"]   = field(default_factory=dict)  # key = source_path
    loaded_keys: set[str]           = field(default_factory=set)
    isas:        dict[str, "ISA"]   = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Insertion helpers
    # ------------------------------------------------------------------ #

    def get_or_create_isa(self, name: str) -> "ISA":
        """Return the shared ISA object for *name*, creating it if needed."""
        from .isa import ISA
        if name not in self.isas:
            self.isas[name] = ISA(name=name)
        return self.isas[name]

    def get_or_create_uarch(self, name: str, isa: "ISA | str", abi: str) -> "Uarch":
        """
        Return the shared Uarch object for *name*, creating it if needed.

        *isa* may be an ISA object or an ISA name; it is always resolved
        through the ISA registry so the object is shared with the
        CompiledTeams targeting it.
        """
        from .isa import ISA as _ISA
        from .uarch import Uarch

        isa_name = isa.name if isinstance(isa, _ISA) else isa
        isa_obj = self.get_or_create_isa(isa_name)

        if name not in self.uarchs:
            self.uarchs[name] = Uarch(name=name, isa=isa_obj, abi=abi)
            return self.uarchs[name]

        existing = self.uarchs[name]
        if existing.isa_name != isa_name:
            # A uarch implements exactly one ISA in the v2 model; a mismatch
            # means two different targets share a simulator name upstream.
            print(f"[Database] Warning: uarch {name!r} already registered with "
                  f"isa={existing.isa_name!r}, ignoring conflicting "
                  f"isa={isa_name!r}")
        elif existing.abi != abi:
            print(f"[Database] Warning: uarch {name!r} already registered with "
                  f"abi={existing.abi!r}, ignoring conflicting abi={abi!r}")
        return existing

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

    def is_loaded(
        self,
        source_path: str,
        uarch_name: str,
        isa_name: str | None = None,
    ) -> bool:
        """
        True if this (seed, uarch[, isa]) combination was already ingested.

        The v1 two-part key is also checked so databases pickled before the
        v2 refactor are not re-ingested.
        """
        if f"{source_path}|{uarch_name}" in self.loaded_keys:
            return True
        if isa_name is not None:
            return f"{source_path}|{uarch_name}|{isa_name}" in self.loaded_keys
        return False

    def mark_loaded(
        self,
        source_path: str,
        uarch_name: str,
        isa_name: str | None = None,
    ) -> None:
        if isa_name is None:
            self.loaded_keys.add(f"{source_path}|{uarch_name}")
        else:
            self.loaded_keys.add(f"{source_path}|{uarch_name}|{isa_name}")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_measurements_for_uarch(
        self, uarch_name: str
    ) -> list[tuple["TPG", int, "CompiledTeam", "TeamMeasurement"]]:
        """
        Return all (tpg, team_id, CompiledTeam, TeamMeasurement) tuples across
        all TPGs for the given uarch — one entry per measured (team, ISA).

        This is the training-set view: each tuple pairs a static feature
        source (the CompiledTeam) with its label (the measurement on that
        uarch).
        """
        result: list[tuple["TPG", int, "CompiledTeam", "TeamMeasurement"]] = []
        for tpg in self.tpgs.values():
            for team in tpg.teams:
                for ct in team.get_compiled_teams_for_uarch(uarch_name):
                    meas = ct.get_measurement(uarch_name)
                    if meas is not None:
                        result.append((tpg, team.id, ct, meas))
        return result

    def get_compiled_teams_for_uarch(
        self, uarch_name: str
    ) -> list[tuple["TPG", int, "CompiledTeam"]]:
        """
        Return all (tpg, team_id, CompiledTeam) triples across all TPGs that
        have a measurement on the given uarch.

        The TPG is included so callers can unambiguously identify which TPG
        each team belongs to (team IDs are only unique within a TPG).
        """
        return [
            (tpg, team_id, ct)
            for tpg, team_id, ct, _meas in self.get_measurements_for_uarch(uarch_name)
        ]

    def get_compiled_teams_for_isa(
        self, isa_name: str
    ) -> list[tuple["TPG", int, "CompiledTeam"]]:
        """
        Return all (tpg, team_id, CompiledTeam) triples compiled for the given
        ISA, measured or not.  Useful for feature extraction, which is
        uarch-independent.
        """
        result: list[tuple["TPG", int, "CompiledTeam"]] = []
        for tpg in self.tpgs.values():
            for team in tpg.teams:
                ct = team.get_compiled_for_isa(isa_name)
                if ct is not None:
                    result.append((tpg, team.id, ct))
        return result

    def all_compiled_teams(self) -> list[tuple["TPG", int, "CompiledTeam"]]:
        """Every (tpg, team_id, CompiledTeam) triple in the database."""
        return [
            (tpg, team.id, ct)
            for tpg in self.tpgs.values()
            for team in tpg.teams
            for ct in team.compiled_teams
        ]

    def isa_of_uarch(self, uarch_name: str) -> "ISA | None":
        """The ISA implemented by *uarch_name*, or None if unknown."""
        uarch = self.uarchs.get(uarch_name)
        return uarch.isa if uarch is not None else None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Database] Saved to {path}  "
              f"({len(self.tpgs)} TPGs, {len(self.isas)} ISAs, "
              f"{len(self.uarchs)} uarchs)")

    @staticmethod
    def load(path: str | Path) -> "Database":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database file not found: {path}")
        with open(path, "rb") as fh:
            db: Database = pickle.load(fh)
        print(f"[Database] Loaded from {path}  "
              f"({len(db.tpgs)} TPGs, {len(db.isas)} ISAs, "
              f"{len(db.uarchs)} uarchs)")
        return db

    def __setstate__(self, state: dict) -> None:
        """
        Restore a pickled Database, migrating v1 files to the v2 model.

        Uarch / CompiledTeam / TeamMeasurement each migrate themselves during
        unpickling (they are reconstructed before this runs).  What is left
        here is to rebuild the ISA registry that v1 databases never had, to
        re-share one ISA instance per name, and to merge the per-uarch
        duplicates v1 created for a single compilation.
        """
        state.setdefault("isas", {})
        self.__dict__.update(state)
        self._rebuild_isa_registry()
        merged = self.consolidate_compiled_teams()
        if merged:
            print(f"[Database] Migrated v1 layout: merged {merged} duplicate "
                  f"compiled team(s) into per-ISA objects carrying several "
                  f"measurements")

    def _rebuild_isa_registry(self) -> None:
        """Register every ISA reachable from uarchs / compiled teams, and make
        objects with the same name identical."""
        from .isa import ISA

        def canonical(isa: "ISA | None") -> "ISA | None":
            if isa is None:
                return None
            if isa.name not in self.isas:
                self.isas[isa.name] = isa
            return self.isas[isa.name]

        for uarch in self.uarchs.values():
            uarch.isa = canonical(uarch.isa) or ISA(name="")
        for _tpg, _tid, ct in self.all_compiled_teams():
            ct.isa = canonical(ct.isa) or ISA(name="")

    def refresh_instructions(self, force: bool = False) -> int:
        """
        Re-parse the instructions of every CompiledTeam whose parser version
        is out of date (see :meth:`CompiledTeam.refresh_instructions`).

        Needed after a change to what the parser records — e.g. keeping the
        callee of a call so that ``jal expf`` and ``jal logf`` count as
        different operations.  The stored disassembly text is the source, so
        no result folder has to be re-read; save the database afterwards to
        make it stick.

        Returns:
            Number of CompiledTeams whose instructions were re-parsed.
        """
        n = sum(
            1
            for _tpg, _tid, ct in self.all_compiled_teams()
            if ct.refresh_instructions(force=force)
        )
        if n:
            print(f"[Database] Re-parsed instructions of {n} compiled team(s) "
                  f"with the current parser (call targets are now kept)")
        return n

    def consolidate_compiled_teams(self) -> int:
        """
        Merge CompiledTeams of the same Team that share an ISA *and* identical
        code into a single object holding all their measurements.

        v1 created one CompiledTeam per uarch, so a v1 database contains as
        many copies of one compilation as there are uarchs running it.  In the
        v2 model that is a single CompiledTeam with several measurements.
        Compilations whose code differs are left alone (they are not the same
        binary, whatever the ISA string says).

        Returns:
            Number of CompiledTeam objects removed by merging.
        """
        removed = 0
        for tpg in self.tpgs.values():
            for team in tpg.teams:
                merged: dict[tuple[str, str], "CompiledTeam"] = {}
                kept: list["CompiledTeam"] = []
                for ct in team.compiled_teams:
                    key = (ct.isa_name, ct.code)
                    first = merged.get(key)
                    if first is None:
                        merged[key] = ct
                        kept.append(ct)
                        continue
                    for uarch_name, meas in ct.measurements.items():
                        first.measurements.setdefault(uarch_name, meas)
                    if first.feature_vector is None:
                        first.feature_vector = ct.feature_vector
                    removed += 1
                team.compiled_teams = kept
        return removed

    # ------------------------------------------------------------------ #
    # Pretty-printing
    # ------------------------------------------------------------------ #

    def _count_measurements_for_uarch(self, uarch_name: str) -> int:
        return sum(
            1
            for tpg in self.tpgs.values()
            for team in tpg.teams
            for ct in team.compiled_teams
            if ct.has_measurement_for(uarch_name)
        )

    # Kept under its v1 name for callers that used it.
    _count_compiled_teams_for_uarch = _count_measurements_for_uarch

    def _count_compiled_teams_for_isa(self, isa_name: str) -> int:
        return sum(
            1
            for tpg in self.tpgs.values()
            for team in tpg.teams
            for ct in team.compiled_teams
            if ct.isa_name == isa_name
        )

    def print_summary(self) -> None:
        total_teams = sum(len(tpg.teams) for tpg in self.tpgs.values())
        total_compiled = sum(
            len(team.compiled_teams)
            for tpg in self.tpgs.values()
            for team in tpg.teams
        )

        print("\n" + "═" * 70)
        print(f"  DATABASE SUMMARY  —  {len(self.tpgs)} TPGs  ·  "
              f"{len(self.isas)} ISAs  ·  {len(self.uarchs)} Uarchs")
        print(f"  ·  Teams (unique): {total_teams}"
              f"  ·  Compiled teams: {total_compiled}")
        print("═" * 70)

        print("\n  ISAs:")
        for isa in self.isas.values():
            n = self._count_compiled_teams_for_isa(isa.name)
            impl = sorted(u.name for u in self.uarchs.values()
                          if u.isa_name == isa.name)
            print(f"    {isa.name}  ({n} compiled teams)"
                  f"  implemented by {impl}")

        print("\n  Uarchs:")
        for uarch in self.uarchs.values():
            n = self._count_measurements_for_uarch(uarch.name)
            print(f"    {uarch.name}  isa={uarch.isa_name}  abi={uarch.abi}"
                  f"  ({n} measured teams)")

        print(f"\n  TPGs ({len(self.tpgs)}):")
        for tpg in self.tpgs.values():
            isa_names = sorted({
                ct.isa_name
                for team in tpg.teams
                for ct in team.compiled_teams
            })
            uarch_names = sorted({
                name
                for team in tpg.teams
                for ct in team.compiled_teams
                for name in ct.uarch_names()
            })
            # Display the human-readable name; source_path shown for traceability
            print(f"    {tpg.name}")
            print(f"      path={tpg.source_path}")
            print(f"      dtype={tpg.dtype}"
                  f"  teams={len(tpg.teams)}"
                  f"  isas={isa_names}"
                  f"  uarchs={uarch_names}")
        print("═" * 70)

        # Traversal / instrumentation-overcost section (own class).
        # Local import keeps the classes↔analysis dependency lazy.
        from analysis.traversal import TraversalAnalyzer
        TraversalAnalyzer.print_summary(self)

        # Dispatch section — only printed when that instrumentation was run.
        from analysis.dispatch import DispatchAnalyzer
        DispatchAnalyzer.print_summary(self)

    def print_uarch(self, uarch_name: str, max_teams: int | None = None) -> None:
        """
        Print every CompiledTeam measured on uarch_name: code, instructions,
        the latency measured on that uarch, and the feature vector (if already
        computed).
        """
        uarch = self.uarchs.get(uarch_name)
        if uarch is None:
            print(f"[Database] Unknown uarch: {uarch_name!r}")
            return

        quads = self.get_measurements_for_uarch(uarch_name)
        if max_teams is not None:
            quads = quads[:max_teams]

        bar = "─" * 70
        print(f"\n{'═'*70}")
        print(f"  UARCH: {uarch.name}  |  ISA: {uarch.isa_name}  |  ABI: {uarch.abi}")
        print(f"  Showing {len(quads)} compiled team(s)")
        print(f"{'═'*70}")

        for tpg, team_id, ct, meas in quads:
            # tpg is the direct owner — no search needed, no misidentification
            print(f"\n{bar}")
            print(f"  Team {team_id}  |  TPG: {tpg.name}  |  path: {tpg.source_path}")
            print(f"  Compiled for ISA: {ct.isa_name}")
            if meas:
                print(f"  Latency : {meas.latency:.2f} cycles"
                      f"  stddev={meas.stddev:.2f}"
                      f"  nb_measurements={meas.nb_measurements}")
            other = [n for n in ct.uarch_names() if n != uarch_name]
            if other:
                print(f"  Also measured on: {other}")
            print(f"  asm:")
            print(ct.code)
            print(f"\n  Instructions ({len(ct.instructions)}):")
            for instr in ct.instructions:
                # `operation` carries the callee for calls (jal_expf, …),
                # which is what the feature names are built from.
                print(f"    {instr.operation:<20} {' '.join(instr.operands)}")

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

    def print_isa(self, isa_name: str, max_teams: int | None = None) -> None:
        """
        Print every CompiledTeam compiled for *isa_name*, with the latency
        measured on each uarch implementing that ISA.
        """
        isa = self.isas.get(isa_name)
        if isa is None:
            print(f"[Database] Unknown ISA: {isa_name!r}")
            return

        triples = self.get_compiled_teams_for_isa(isa_name)
        if max_teams is not None:
            triples = triples[:max_teams]

        bar = "─" * 70
        print(f"\n{'═'*70}")
        print(f"  ISA: {isa.name}")
        print(f"  Showing {len(triples)} compiled team(s)")
        print(f"{'═'*70}")

        for tpg, team_id, ct in triples:
            print(f"\n{bar}")
            print(f"  Team {team_id}  |  TPG: {tpg.name}")
            print(f"  Instructions: {len(ct.instructions)}")
            if ct.measurements:
                print(f"  Measurements ({len(ct.measurements)}):")
                for name, m in ct.measurements.items():
                    print(f"    {name:<32} {m.latency:>10.2f} cycles"
                          f"  stddev={m.stddev:.2f}  n={m.nb_measurements}")
            else:
                print("  Measurements: none")
        print(bar)