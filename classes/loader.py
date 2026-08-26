"""
classes/loader.py — discovers and ingests TPG result folders into a Database.

Folder layout expected::

    <root>/
    └── training_results/
        ├── <seed_dir_0>/
        │   └── inference/
        │       └── results/
        │           ├── <uarch_isa_abi_dtype>/
        │           │   ├── latencies.json
        │           │   └── disassembly_tpg_inference_instrTeams_instrTPG.txt
        │           └── <uarch2_isa_abi_dtype>/
        │               ├── latencies.json
        │               └── disassembly_tpg_inference_instrTeams_instrTPG.txt
        ├── <seed_dir_1>/
        │   └── …
        └── …

One seed directory = one TPG.  Each subdirectory of ``inference/results/``
is one (uarch × ISA) run: it has its own disassembly and latency JSON.

v2 ingestion model
------------------
The disassembly of a run describes code compiled for the run's *ISA*, while
the JSON latencies describe how that code behaved on the run's *uarch*.  So
each uarch directory produces, per team:

  * one CompiledTeam per ISA — created the first time that ISA is seen for
    the team, then **reused** by every later uarch implementing the same ISA;
  * one TeamMeasurement attached to that CompiledTeam under the uarch name.

Two directories that differ only by uarch therefore share a single
CompiledTeam (one code/instruction/feature copy) carrying two measurements.

Identity vs. display
--------------------
``source_path``  — absolute path of the seed directory, used as the unique
                   key in Database.tpgs and in loaded_keys.  Guaranteed to
                   be unambiguous across multiple roots and across sessions
                   on the same machine.
``display_name`` — seed-directory basename, used only for human-readable
                   log output and print_summary / print_uarch output.
"""

from __future__ import annotations

from pathlib import Path

from .database import Database
from .tpg import TPG, ClassLatencyPair
from .team import Team, CompiledTeam
from .teamMeasurements import TeamMeasurement
from analysis.disassembler import Disassembler, TeamBlock, TPGLatencyData
from analysis.traversal import TraversalAnalyzer

_DISASM_FILENAME = "disassembly_tpg_inference_instrTeams_instrTPG.txt"


class Loader:
    """
    Scans a model root directory and loads all TPG results into a Database.

    All methods are static — Loader is a namespace, not an instance.
    """

    @staticmethod
    def load_folder(root: str | Path, db: Database | None = None) -> Database:
        """
        Recursively find every ``inference/results/`` directory under *root*,
        iterate its uarch subdirectories, and populate a Database.

        Already-loaded (source_path, uarch, isa) combinations are silently
        skipped.

        Args:
            root: Top-level model directory to scan.
            db:   Existing Database to extend; a new one is created if None.

        Returns:
            The populated (or extended) Database.
        """
        if db is None:
            db = Database()

        root = Path(root)
        results_dirs = sorted(root.glob("**/inference/results"))

        if not results_dirs:
            print(f"[Loader] No 'inference/results' directories found under {root}")
            return db

        print(f"[Loader] Found {len(results_dirs)} "
              f"result director{'y' if len(results_dirs) == 1 else 'ies'}")

        for results_dir in results_dirs:
            Loader._load_results_dir(results_dir, db)

        return db

    @staticmethod
    def _load_results_dir(results_dir: Path, db: Database) -> None:
        """
        Load one ``inference/results/`` directory by iterating its
        uarch subdirectories.

        ``source_path`` is the resolved absolute path of the seed directory
        (two levels above ``inference/results/``).  It is the unique TPG key.
        ``display_name`` is the seed-directory basename, used only for logs.
        """
        seed_dir     = results_dir.parent.parent
        source_path  = str(seed_dir.resolve())   # unique identity key
        display_name = seed_dir.name              # human-readable, display only

        uarch_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())

        if not uarch_dirs:
            print(f"[Loader]   SKIP {display_name}: no uarch subdirs in results/")
            return

        for uarch_dir in uarch_dirs:
            Loader._load_uarch_dir(uarch_dir, source_path, display_name, db)

    @staticmethod
    def _load_uarch_dir(
        uarch_dir:    Path,
        source_path:  str,   # absolute seed-dir path — unique TPG key
        display_name: str,   # seed-dir basename — for log output only
        db:           Database,
    ) -> None:
        """
        Load one ``<uarch_isa_abi_dtype>/`` subdirectory.

        Parses the local disassembly and latency JSON, then, for each team,
        finds (or creates) the CompiledTeam for this run's ISA and attaches
        the measurement taken on this run's uarch.

        Args:
            uarch_dir:    Path to the uarch subdirectory.
            source_path:  Absolute path of the seed dir (unique TPG key).
            display_name: Basename of the seed dir (for log output only).
            db:           Database to populate.
        """
        json_path   = uarch_dir / "latencies.json"
        disasm_path = uarch_dir / _DISASM_FILENAME

        if not json_path.exists():
            print(f"[Loader]   SKIP {uarch_dir.name}: missing latencies.json")
            return
        if not disasm_path.exists():
            print(f"[Loader]   SKIP {uarch_dir.name}: missing disassembly file")
            return

        lat_data: TPGLatencyData = Disassembler.parse_latency_json(str(json_path))
        uarch_name = lat_data.simulator
        isa_name   = lat_data.isa

        # Guard: directory name must be "<uarch>_rv32<isa>_<abi>_<dtype>".
        # The uarch prefix (everything before the first "_rv32" token) must
        # match the simulator name reported inside latencies.json.
        if "_rv32" not in uarch_dir.name:
            raise ValueError(
                f"Unexpected directory name format (no '_rv32' token): {uarch_dir.name}"
            )
        dir_uarch_prefix = uarch_dir.name.split("_rv32")[0]
        if dir_uarch_prefix != uarch_name:
            raise ValueError(
                f"Simulator name mismatch in {uarch_dir}:\n"
                f"  directory    : {uarch_dir.name}\n"
                f"  latencies.json: {uarch_name}"
            )

        # Dedup check uses source_path (not display_name) as the TPG key, and
        # now the ISA too — the same simulator may appear for several ISAs.
        if db.is_loaded(source_path, uarch_name, isa_name):
            print(f"[Loader]   SKIP (already loaded): {display_name} × "
                  f"{uarch_name} × {isa_name}")
            return

        blocks: dict[int, TeamBlock] = Disassembler.parse_tpg_file(str(disasm_path))

        isa = db.get_or_create_isa(isa_name)
        uarch = db.get_or_create_uarch(
            name=uarch_name,
            isa=isa,
            abi=lat_data.abi,
        )
        tpg = db.get_or_create_tpg(
            source_path=source_path,   # unique key
            name=display_name,         # display only
            dtype=lat_data.dtype,
        )

        # ── Graph traversals (LE_states.h) ─────────────────────────────
        # Parsed once per TPG; the mapping is ISA/uarch-independent.  It
        # lives at <seed_dir>/outLogs/precalcul/LE_states.h.
        if not tpg.traversals:
            le_path = TraversalAnalyzer.le_states_path_for(source_path)
            tpg.traversals = TraversalAnalyzer.parse_le_states(le_path)
            if tpg.traversals:
                print(f"[Loader]   parsed {len(tpg.traversals)} traversal(s) "
                      f"from LE_states.h")
            else:
                print(f"[Loader]   note: no LE_states.h traversal map found "
                      f"under {le_path.parent}")

        # ── Whole-TPG class latencies for both instrumentation modes ───
        # Keyed by uarch: these are timing figures, not compilation data.
        tpg.class_latencies[uarch_name] = {
            cid: ClassLatencyPair(
                class_id=cid,
                tpg_only=lat_data.classes_tpg_only.get(cid).avg_cycles
                    if lat_data.classes_tpg_only.get(cid) else 0.0,
                tpg_teams=teams_cl.avg_cycles,
                tpg_only_stddev=lat_data.classes_tpg_only.get(cid).stddev_cycles
                    if lat_data.classes_tpg_only.get(cid) else 0.0,
                tpg_teams_stddev=teams_cl.stddev_cycles,
            )
            for cid, teams_cl in lat_data.classes_tpg_teams.items()
        }

        added = reused = skipped_no_lat = code_mismatch = 0

        for team_id, block in sorted(blocks.items()):
            tl = lat_data.get_team_latency(team_id)
            if tl is None:
                skipped_no_lat += 1
                continue

            # Get or create the Team (identity node, ISA/uarch-agnostic).
            # Team IDs are only unique within a TPG — the tpg object here
            # is always the correct one because it was fetched by source_path.
            team = tpg.get_team(team_id)
            if team is None:
                team = Team(id=team_id)
                tpg.add_team(team)

            # Get or create the CompiledTeam for this ISA.  A second uarch
            # implementing the same ISA reuses the existing one and only
            # contributes an extra measurement.
            ct = team.get_compiled_for_isa(isa)
            if ct is None:
                # feature_vector is intentionally left None at load time.
                # It is computed at training time so feature extraction rules
                # can be changed freely without reloading the database.
                ct = CompiledTeam(
                    isa=isa,
                    code=block.code,
                    instructions=block.instructions,
                )
                team.add_compiled_team(ct)
                added += 1
            else:
                reused += 1
                if ct.code != block.code:
                    # Same ISA should mean identical code; if it does not,
                    # keep the first version and flag it rather than silently
                    # mixing two binaries under one CompiledTeam.
                    code_mismatch += 1

            meas = TeamMeasurement(
                latency=tl.avg_cycles,
                nb_measurements=tl.nb_measurements,
                stddev=tl.stddev_cycles,
                uarch=uarch,
            )
            ct.add_measurement(meas, uarch=uarch)

        db.mark_loaded(source_path, uarch_name, isa_name)
        print(
            f"[Loader]   + {display_name}"
            f"\n             uarch={uarch_name}  isa={isa_name}"
            f"  new compiled teams={added}"
            f"  reused={reused}"
            f"  measurements added={added + reused}"
            f"  skipped={skipped_no_lat} (no latency data)"
        )
        if code_mismatch:
            print(f"[Loader]   Warning: {code_mismatch} team(s) had different "
                  f"disassembly for isa={isa_name} than the already-loaded "
                  f"compilation; kept the first one.")