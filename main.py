"""
main.py — CLI for the TPG latency-estimation pipeline.

Usage
-----
    python main.py load <model_root> [--db <path>] [--save <path>]
    python main.py load <model_root> --db <existing.pkl> --save <existing.pkl>
    python main.py summary --db <path>
    python main.py inspect --db <path> --uarch <uarch_name> [--max-teams N]
    python main.py inspect --db <path> --isa  <isa_name>   [--max-teams N]
    python main.py train --db <path> --uarch <uarch_name> [--test-size 0.2] [--seed 42]

Model note (v2)
---------------
Features are extracted from a CompiledTeam, which is compiled for one ISA;
latency labels come from the TeamMeasurement taken on the requested uarch.
Training therefore pairs, for every measured team, the ISA-level feature
vector with the uarch-level latency.
"""

from __future__ import annotations

import argparse
import sys

from classes.database import Database
from classes.loader import Loader
from analysis.analyzer import FeaturesAnalyzer
from analysis.regression import Regressor


def cmd_load(args: argparse.Namespace) -> None:
    if args.db and __import__("pathlib").Path(args.db).exists():
        db = Database.load(args.db)
    else:
        db = Database()

    db = Loader.load_folder(args.root, db)
    db.print_summary()

    save_path = args.save or args.db
    if save_path:
        db.save(save_path)


def cmd_summary(args: argparse.Namespace) -> None:
    db = Database.load(args.db)
    db.print_summary()


def _compute_features(compiled_teams) -> None:
    """
    Compute feature vectors for the given CompiledTeams.

    Features are static and ISA-determined, so they are computed once per
    CompiledTeam and shared by every uarch measurement attached to it.
    """
    seen: set[int] = set()
    for ct in compiled_teams:
        if id(ct) in seen:
            continue
        seen.add(id(ct))
        ct.feature_vector = FeaturesAnalyzer.analyze_instructions(ct.instructions)


def _compute_features_for_uarch(db: Database, uarch_name: str) -> None:
    _compute_features(ct for _tpg, _tid, ct
                      in db.get_compiled_teams_for_uarch(uarch_name))


def _compute_features_for_isa(db: Database, isa_name: str) -> None:
    _compute_features(ct for _tpg, _tid, ct
                      in db.get_compiled_teams_for_isa(isa_name))


def cmd_inspect(args: argparse.Namespace) -> None:
    if not args.uarch and not args.isa:
        print("[inspect] Provide --uarch or --isa")
        sys.exit(1)

    db = Database.load(args.db)

    if args.isa:
        _compute_features_for_isa(db, args.isa)
        db.print_isa(args.isa, max_teams=args.max_teams)
    else:
        _compute_features_for_uarch(db, args.uarch)
        db.print_uarch(args.uarch, max_teams=args.max_teams)


def cmd_train(args: argparse.Namespace) -> None:
    db = Database.load(args.db)

    # One sample per (CompiledTeam, measurement-on-this-uarch) pair.
    quads = db.get_measurements_for_uarch(args.uarch)
    if len(quads) < 2:
        print(f"[train] Need at least 2 measured teams for {args.uarch!r}, "
              f"found {len(quads)}")
        sys.exit(1)

    # ── Feature extraction — done here so rules can be freely changed ──
    _compute_features(ct for _tpg, _tid, ct, _meas in quads)

    # ── Build training data ────────────────────────────────────────────
    feature_vectors = [ct.feature_vector for _tpg, _tid, ct, _meas in quads]
    latencies       = [meas.latency      for _tpg, _tid, _ct, meas in quads]

    uarch = db.uarchs.get(args.uarch)
    isa_str = f"  isa={uarch.isa_name!r}" if uarch else ""
    print(f"[train] {len(feature_vectors)} samples  uarch={args.uarch!r}{isa_str}")
    model = Regressor.train(
        feature_vectors,
        latencies,
        test_size=args.test_size,
        random_state=args.seed,
    )
    model.print_report()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="TPG latency-estimation pipeline",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="Ingest a model folder into the database")
    p_load.add_argument("root", help="Model root directory to scan")
    p_load.add_argument("--db",   default=None, help="Existing database to extend")
    p_load.add_argument("--save", default=None,
                        help="Where to save the updated database (defaults to --db)")

    p_sum = sub.add_parser("summary", help="Print database summary")
    p_sum.add_argument("--db", required=True, help="Database file")

    p_ins = sub.add_parser("inspect",
                           help="Print all compiled teams for a uarch or an ISA")
    p_ins.add_argument("--db",        required=True, help="Database file")
    p_ins.add_argument("--uarch",     default=None, help="Uarch name (simulator field)")
    p_ins.add_argument("--isa",       default=None, help="ISA name (isa field)")
    p_ins.add_argument("--max-teams", type=int, default=None,
                       help="Limit number of teams printed")

    p_tr = sub.add_parser("train", help="Train a Lasso model for one uarch")
    p_tr.add_argument("--db",        required=True, help="Database file")
    p_tr.add_argument("--uarch",     required=True, help="Uarch name to train on")
    p_tr.add_argument("--test-size", type=float, default=0.20,
                      help="Fraction held out for testing (default: 0.20)")
    p_tr.add_argument("--seed",      type=int,   default=42,
                      help="Random seed for the train/test split (default: 42)")

    args = parser.parse_args()

    dispatch = {
        "load":    cmd_load,
        "summary": cmd_summary,
        "inspect": cmd_inspect,
        "train":   cmd_train,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()