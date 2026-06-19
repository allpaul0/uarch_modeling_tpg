"""
main.py — CLI for the TPG latency-estimation pipeline.

Usage
-----
    python main.py load <model_root> [--db <path>] [--save <path>]
    python main.py load <model_root> --db <existing.pkl> --save <existing.pkl>
    python main.py summary --db <path>
    python main.py inspect --db <path> --uarch <uarch_name> [--max-teams N]
    python main.py train --db <path> --uarch <uarch_name> [--test-size 0.2] [--seed 42]
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


def _compute_features(db: Database, uarch_name: str) -> None:
    """Compute feature vectors for all CompiledTeams targeting uarch_name."""
    # get_compiled_teams_for_uarch now returns (tpg, team_id, ct) triples.
    for _tpg, _team_id, ct in db.get_compiled_teams_for_uarch(uarch_name):
        ct.feature_vector = FeaturesAnalyzer.analyze_instructions(ct.instructions)
        

def cmd_inspect(args: argparse.Namespace) -> None:
    db = Database.load(args.db)
    _compute_features(db, args.uarch)
    db.print_uarch(args.uarch, max_teams=args.max_teams)


def cmd_train(args: argparse.Namespace) -> None:
    db = Database.load(args.db)

    triples = db.get_compiled_teams_for_uarch(args.uarch)
    if len(triples) < 2:
        print(f"[train] Need at least 2 compiled teams for {args.uarch!r}, "
              f"found {len(triples)}")
        sys.exit(1)

    # ── Feature extraction — done here so rules can be freely changed ──
    _compute_features(db, args.uarch)

    # ── Build training data ────────────────────────────────────────────
    feature_vectors = [ct.feature_vector for _tpg, _tid, ct in triples]
    latencies       = [ct.measurement.latency for _tpg, _tid, ct in triples]

    print(f"[train] {len(feature_vectors)} samples  uarch={args.uarch!r}")
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

    p_ins = sub.add_parser("inspect", help="Print all compiled teams for a given uarch")
    p_ins.add_argument("--db",        required=True, help="Database file")
    p_ins.add_argument("--uarch",     required=True, help="Uarch name (simulator field)")
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