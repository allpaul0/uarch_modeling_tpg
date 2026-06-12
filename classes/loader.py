"""
classes/loader.py — discovers and ingests TPG result folders into a Database.

Folder layout expected::

    <root>/
    └── training_results/
        ├── <seed_dir_0>/
        │   └── inference/
        │       └── results/
        │           ├── disassembly_tpg_inference_instrTeams_instrTPG.txt
        │           ├── <uarch1>_<isa>_<abi>_<dtype>.json
        │           └── <uarch2>_<isa>_<abi>_<dtype>.json   ← optional 2nd uarch
        ├── <seed_dir_1>/
        │   └── …
        └── …

One seed directory = one TPG.  The seed-dir basename is used as the TPG name.
Each JSON file in ``inference/results/`` corresponds to one Uarch.  All JSON
files share the same disassembly text file (same binary, different simulators).
"""

from __future__ import annotations

from pathlib import Path

from .database import Database
from .tpg import TPG
from .team import Team
from .teamMeasurements import TeamMeasurement
from .features import FeatureVector
from analysis.disassembler import Disassembler, TeamBlock, TPGLatencyData
from analysis.analyzer import FeaturesAnalyzer

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
        parse its disassembly and JSON files, and populate a Database.

        Already-loaded (TPG, Uarch) pairs are silently skipped (deduplication).

        Args:
            root: Top-level model directory
                  (e.g. ``model_tpg_l2e2_zmmul_compbare_compExpAr``).
            db:   An existing Database to extend.  If None, a new one is
                  created.

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

        print(f"[Loader] Found {len(results_dirs)} result director{'y' if len(results_dirs)==1 else 'ies'}")

        for results_dir in results_dirs:
            Loader._load_results_dir(results_dir, db)

        return db

    @staticmethod
    def _load_results_dir(results_dir: Path, db: Database) -> None:
        """
        Load one ``inference/results/`` directory.

        The TPG name is the seed-directory basename (two levels above
        ``inference/results/``).
        """
        # seed_dir is two levels up: results_dir / .. / .. = seed_dir
        seed_dir = results_dir.parent.parent
        tpg_name = seed_dir.name

        disasm_path = results_dir / _DISASM_FILENAME
        if not disasm_path.exists():
            print(f"[Loader]   SKIP {tpg_name}: no disassembly file")
            return

        json_files = sorted(results_dir.glob("*.json"))
        if not json_files:
            print(f"[Loader]   SKIP {tpg_name}: no JSON files")
            return

        # Parse disassembly once — shared by all JSON files in this dir
        blocks: dict[int, TeamBlock] = Disassembler.parse_tpg_file(str(disasm_path))

        for json_path in json_files:
            Loader._load_json(json_path, tpg_name, blocks, db)

    @staticmethod
    def _load_json(
        json_path: Path,
        tpg_name: str,
        blocks: dict[int, TeamBlock],
        db: Database,
    ) -> None:
        """
        Parse one JSON latency file and merge its teams into the Database.

        If the (tpg_name, uarch_name) pair is already loaded, this is a
        no-op.
        """
        lat_data: TPGLatencyData = Disassembler.parse_latency_json(str(json_path))
        uarch_name = lat_data.simulator

        if db.is_loaded(tpg_name, uarch_name):
            print(f"[Loader]   SKIP (already loaded): {tpg_name} × {uarch_name}")
            return

        uarch = db.get_or_create_uarch(
            name=uarch_name,
            isa=lat_data.isa,
            abi=lat_data.abi,
        )
        tpg = db.get_or_create_tpg(
            name=tpg_name,
            dtype=lat_data.dtype,
        )

        added = skipped_no_lat = 0

        for team_id, block in sorted(blocks.items()):
            tl = lat_data.get_team_latency(team_id)
            if tl is None:
                skipped_no_lat += 1
                continue

            # Re-use an existing Team object if this TPG was already partially
            # populated by a previous uarch's JSON (same binary → same instrs).
            team = tpg.get_team(team_id)
            if team is None:
                team = Team(id=team_id, code=block.code)
                team.set_instructions(block.instructions)
                fv = FeaturesAnalyzer.analyze_instructions(
                    block.instructions, id_team=team_id
                )
                team.set_feature_vector(fv)
                tpg.add_team(team)

            meas = TeamMeasurement(
                uarch=uarch,
                id_team=team_id,
                latency=tl.avg_cycles,
                nb_measurements=tl.nb_measurements,
                stddev=tl.stddev_cycles,
                cv=tl.coefficient_variation,
            )
            team.add_measurement(meas)
            uarch.add_measurement(meas)
            added += 1

        db.mark_loaded(tpg_name, uarch_name)
        print(
            f"[Loader]   + {tpg_name[:60]}"
            f"\n             uarch={uarch_name}"
            f"  added={added} teams"
            f"  skipped={skipped_no_lat} zero-latency"
        )