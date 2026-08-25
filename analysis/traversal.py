"""
analysis/traversal.py — per-traversal instrumentation-overcost analysis.

This module owns *all* the work requested on top of the base pipeline:

1.  Parse ``outLogs/precalcul/LE_states.h`` to recover, for every TPG, which
    Teams are visited by each graph traversal (a.k.a. class / learning
    example)::

        // [0] -> [T0 -> T1 -> T2 -> T7 -> T9 -> T6]

2.  Quantify the *instrumentation overcost* of each traversal — the extra
    cycles introduced by adding per-team timing probes.  ``latencies.json``
    holds two whole-TPG benchmarks per class:

        instrTPG            → only the TPG is instrumented
        instrTeams_instrTPG → the TPG *and* every team are instrumented

    overcost(class) = AvgCyclesPerClass[instrTeams_instrTPG]
                    - AvgCyclesPerClass[instrTPG]

3.  Sum the individual measured Team latencies along each traversal and put
    that "Σ team latency" next to the two whole-TPG figures, so the three
    measures can be compared traversal by traversal.

Everything here reads data already stored on the model (``TPG.traversals``,
``TPG.class_latencies`` and the per-team ``CompiledTeam.measurement``), so it
works purely from a loaded/pickled Database — no re-reading of source files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes.tpg import TPG
    from classes.database import Database


# One "// [k] -> [T.. -> T.. -> ..]" mapping line in LE_states.h.
_TRAVERSAL_RE = re.compile(r"\[(\d+)\]\s*->\s*\[([^\]]*)\]")
_TEAM_RE = re.compile(r"T(\d+)")

# Location of the traversal-mapping header relative to a seed directory.
_LE_STATES_RELPATH = ("outLogs", "precalcul", "LE_states.h")


@dataclass
class TraversalReport:
    """
    The three latency measures for one traversal of one (TPG, uarch), plus
    the derived instrumentation overcost.

    Attributes
    ----------
    traversal_id:      Class / traversal id (from LE_states.h).
    teams:             Ordered team ids visited by the traversal.
    tpg_only_latency:  Whole-TPG cycles, TPG-only instrumentation.
    tpg_teams_latency: Whole-TPG cycles, TPG + team instrumentation.
    team_latency_sum:  Σ of measured AvgCyclesPerTeam over ``teams``.
    missing_teams:     Teams in the traversal with no measured latency
                       (excluded from ``team_latency_sum``).
    """
    traversal_id: int
    teams: list[int]
    tpg_only_latency: float
    tpg_teams_latency: float
    team_latency_sum: float
    missing_teams: list[int] = field(default_factory=list)

    @property
    def instrumentation_overcost(self) -> float:
        """Extra cycles caused by the per-team probes (absolute)."""
        return self.tpg_teams_latency - self.tpg_only_latency

    @property
    def overcost_pct(self) -> float:
        """Instrumentation overcost as a percentage of the TPG-only latency."""
        if self.tpg_only_latency == 0.0:
            return 0.0
        return self.instrumentation_overcost / self.tpg_only_latency * 100.0

    @property
    def graph_overhead(self) -> float:
        """
        TPG-only latency in excess of the summed team latencies (cycles).

        ``tpg_only_latency − team_latency_sum`` — the cost of traversing the
        graph itself (edge/program evaluation and jumps between teams) that is
        not accounted for by any team's measured body.
        """
        return self.tpg_only_latency - self.team_latency_sum

    @property
    def graph_overhead_pct(self) -> float:
        """
        Graph overhead as a percentage of the summed team latencies, i.e. how
        much the graph traversal adds on top of raw team compute.  Base is the
        team sum (parallel to the instrumentation overcost, which is expressed
        relative to its own baseline, the TPG-only latency).
        """
        if self.team_latency_sum == 0.0:
            return 0.0
        return self.graph_overhead / self.team_latency_sum * 100.0


class TraversalAnalyzer:
    """
    Computes and prints per-traversal instrumentation overcost and team
    latency sums.  All methods are static — this is a namespace.
    """

    # ------------------------------------------------------------------ #
    # LE_states.h parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_le_states(path: str | Path) -> dict[int, list[int]]:
        """
        Parse an ``LE_states.h`` file and return ``{traversal_id: [team_ids]}``.

        Only the ``// [k] -> [T.. -> T..]`` comment mapping is read; the
        numeric seed tables below it are ignored.  Returns an empty dict if
        the file is missing or contains no mapping.
        """
        path = Path(path)
        if not path.exists():
            return {}

        traversals: dict[int, list[int]] = {}
        for line in path.read_text().splitlines():
            m = _TRAVERSAL_RE.search(line)
            if not m:
                continue
            tid = int(m.group(1))
            teams = [int(t) for t in _TEAM_RE.findall(m.group(2))]
            if teams:
                traversals[tid] = teams
        return traversals

    @staticmethod
    def le_states_path_for(source_path: str | Path) -> Path:
        """Return the expected LE_states.h path for a seed directory."""
        return Path(source_path).joinpath(*_LE_STATES_RELPATH)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _team_latency(tpg: "TPG", team_id: int, uarch_name: str) -> float | None:
        """Measured AvgCyclesPerTeam for one team on one uarch, or None."""
        team = tpg.get_team(team_id)
        if team is None:
            return None
        ct = team.get_compiled_for_uarch(uarch_name)
        if ct is None or ct.measurement is None:
            return None
        return ct.measurement.latency

    @staticmethod
    def analyze_tpg(tpg: "TPG", uarch_name: str) -> list[TraversalReport]:
        """
        Build one :class:`TraversalReport` per traversal for ``(tpg, uarch)``.

        Requires ``tpg.class_latencies[uarch_name]`` (whole-TPG class figures)
        and ``tpg.traversals`` (team membership).  Traversals present in
        either source are reported; a missing counterpart contributes 0.
        """
        class_lat = tpg.class_latencies.get(uarch_name, {})
        traversals = tpg.traversals

        reports: list[TraversalReport] = []
        for tid in sorted(set(class_lat) | set(traversals)):
            teams = traversals.get(tid, [])
            pair = class_lat.get(tid)

            total = 0.0
            missing: list[int] = []
            for team_id in teams:
                lat = TraversalAnalyzer._team_latency(tpg, team_id, uarch_name)
                if lat is None:
                    missing.append(team_id)
                else:
                    total += lat

            reports.append(
                TraversalReport(
                    traversal_id=tid,
                    teams=teams,
                    tpg_only_latency=pair.tpg_only if pair else 0.0,
                    tpg_teams_latency=pair.tpg_teams if pair else 0.0,
                    team_latency_sum=total,
                    missing_teams=missing,
                )
            )
        return reports

    # ------------------------------------------------------------------ #
    # Pretty-printing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_tpg_uarch(tpg: "TPG", uarch_name: str) -> None:
        reports = TraversalAnalyzer.analyze_tpg(tpg, uarch_name)
        if not reports:
            return

        print(f"\n  TPG: {tpg.name}")
        print(f"    uarch: {uarch_name}")

        if not tpg.traversals:
            print("    (no LE_states.h traversal map — team-latency sums "
                  "unavailable; overcost shown per class)")

        header = (
            f"    {'trav':>4}  {'teams':<24} "
            f"{'TPGonly':>8} {'TPG+Tm':>8} "
            f"{'instrOvc':>9} {'(%)':>7}  "
            f"{'ΣteamLat':>9}  {'TPGovh':>8} {'(%)':>7}"
        )
        print(header)
        print("    " + "-" * (len(header) - 4))

        sum_only = sum_teams = sum_over = 0.0
        sum_teamlat = sum_only_wt = 0.0   # accumulate team-derived cols only where teams exist
        for r in reports:
            path_str = "→".join(f"T{t}" for t in r.teams) if r.teams else "—"
            if len(path_str) > 24:
                path_str = path_str[:21] + "..."

            if r.teams:
                teamlat_str = f"{r.team_latency_sum:>9.1f}"
                ovh_str     = f"{r.graph_overhead:>+8.1f}"
                ovhpct_str  = f"{r.graph_overhead_pct:>+6.1f}%"
                sum_teamlat += r.team_latency_sum
                sum_only_wt += r.tpg_only_latency
            else:
                teamlat_str = f"{'—':>9}"
                ovh_str     = f"{'—':>8}"
                ovhpct_str  = f"{'':>7}"

            print(
                f"    {r.traversal_id:>4}  {path_str:<24} "
                f"{r.tpg_only_latency:>8.1f} {r.tpg_teams_latency:>8.1f} "
                f"{r.instrumentation_overcost:>+9.1f} {r.overcost_pct:>+6.1f}%  "
                f"{teamlat_str}  {ovh_str} {ovhpct_str}"
            )
            if r.missing_teams:
                print(f"         └ no measured latency for teams "
                      f"{r.missing_teams} (excluded from Σ / TPGovh)")
            sum_only  += r.tpg_only_latency
            sum_teams += r.tpg_teams_latency
            sum_over  += r.instrumentation_overcost

        instr_pct = (sum_over / sum_only * 100.0) if sum_only else 0.0
        sum_ovh = sum_only_wt - sum_teamlat
        ovh_pct = (sum_ovh / sum_teamlat * 100.0) if sum_teamlat else 0.0

        if sum_teamlat:
            tot_teamlat_str = f"{sum_teamlat:>9.1f}"
            tot_ovh_str     = f"{sum_ovh:>+8.1f}"
            tot_ovhpct_str  = f"{ovh_pct:>+6.1f}%"
        else:
            tot_teamlat_str = f"{'—':>9}"
            tot_ovh_str     = f"{'—':>8}"
            tot_ovhpct_str  = f"{'':>7}"

        print("    " + "-" * (len(header) - 4))
        print(
            f"    {'Σ':>4}  {'(all traversals)':<24} "
            f"{sum_only:>8.1f} {sum_teams:>8.1f} "
            f"{sum_over:>+9.1f} {instr_pct:>+6.1f}%  "
            f"{tot_teamlat_str}  {tot_ovh_str} {tot_ovhpct_str}"
        )

    @staticmethod
    def print_summary(db: "Database") -> None:
        """
        Print the traversal / instrumentation-overcost section for every
        (TPG, uarch) in the database.  Called from ``Database.print_summary``.
        """
        # Which uarchs is each TPG actually compiled for?
        any_output = False
        blocks: list[str] = []
        print("\n" + "═" * 70)
        print("  TRAVERSAL / INSTRUMENTATION-OVERCOST ANALYSIS")
        print("    instrOvc = (TPG+Teams instrumented) − (TPG-only)   "
              "[team-probe cost, % of TPGonly]")
        print("    ΣteamLat = sum of measured per-team latencies along traversal")
        print("    TPGovh   = (TPG-only) − ΣteamLat                   "
              "[graph-traversal overhead, % of ΣteamLat]")
        print("═" * 70)

        for tpg in db.tpgs.values():
            uarch_names = sorted(tpg.class_latencies.keys())
            if not uarch_names:
                # Fall back to uarchs seen on compiled teams (no class data).
                uarch_names = sorted({
                    ct.uarch.name
                    for team in tpg.teams
                    for ct in team.compiled_teams
                })
            for uarch_name in uarch_names:
                if uarch_name in tpg.class_latencies or tpg.traversals:
                    TraversalAnalyzer._print_tpg_uarch(tpg, uarch_name)
                    any_output = True

        if not any_output:
            print("\n  (no class-latency data loaded — re-run 'load' to "
                  "populate traversal/overcost information)")
        print("═" * 70)