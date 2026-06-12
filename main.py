"""
main.py — end-to-end demonstration of the TPG latency-estimation pipeline.

Usage
-----
    python main.py                              # uses synthetic data
    python main.py <disasm.txt> <results.json>  # parse real objdump + JSON
"""

from __future__ import annotations
import sys

from classes import TPG, Team, Instruction, Uarch, TeamMeasurement, FeatureVector
from analysis import Disassembler, FeaturesAnalyzer, Regressor
from analysis.disassembler import TPGLatencyData, TeamBlock


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------

def print_dataset_summary(tpg: TPG) -> None:
    """
    Print a compact table of every team in the TPG with its latency,
    measurement count, instruction count, and first few mnemonics.
    Useful for quick sanity-checks of the loaded dataset.
    """
    header = (
        f"{'ID':>4}  {'#instr':>6}  {'latency':>9}  "
        f"{'stddev':>7}  {'#runs':>6}  {'first mnemonics'}"
    )
    print("\n" + "=" * len(header))
    print(f"Dataset: {tpg.name}  ({len(tpg.teams)} teams)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for team in sorted(tpg.teams, key=lambda t: t.id):
        meas   = team.measurements[0] if team.measurements else None
        lat    = f"{meas.latency:.1f}" if meas else "—"
        stddev = f"{meas.stddev:.2f}"  if meas else "—"
        runs   = str(meas.nb_measurements) if meas else "—"
        mnems  = " ".join(i.mnemonic for i in team.instructions[:6])
        if len(team.instructions) > 6:
            mnems += " …"
        print(
            f"{team.id:>4}  {len(team.instructions):>6}  {lat:>9}  "
            f"{stddev:>7}  {runs:>6}  {mnems}"
        )

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Team builders
# ---------------------------------------------------------------------------

def build_team_from_block(
    block: TeamBlock,
    uarch: Uarch,
    nb_measurements: int,
    avg_latency: float,
    stddev: float = 0.0,
    cv: float = 0.0,
) -> Team:
    """
    Build a fully-populated Team from a parsed :class:`TeamBlock` and the
    measured latency data from the JSON file.

    Args:
        block:           Parsed disassembly block (instructions + raw code).
        uarch:           Micro-architecture the measurement was taken on.
        nb_measurements: Number of individual runs (Count in the JSON).
        avg_latency:     Average measured cycle count (AvgCyclesPerTeam).
        stddev:          Standard deviation of cycles (StddevCyclesPerTeam).
        cv:              Coefficient of variation (CoefficientVariation).

    Returns:
        A fully populated Team with code, instructions, feature vector,
        and measurement attached.
    """
    team = Team(id=block.team_id, code=block.code)
    team.set_instructions(block.instructions)

    fv = FeaturesAnalyzer.analyze_instructions(block.instructions, id_team=block.team_id)
    team.set_feature_vector(fv)

    meas = TeamMeasurement(
        uarch=uarch,
        id_team=block.team_id,
        latency=avg_latency,
        nb_measurements=nb_measurements,
        stddev=stddev,
        cv=cv,
    )
    team.add_measurement(meas)
    uarch.add_measurement(meas)

    return team


def build_tpg_from_files(
    disasm_path: str,
    json_path: str,
) -> tuple[TPG, Uarch]:
    """
    Parse a TPG disassembly file and its companion JSON latency file, then
    build a fully-populated TPG.

    Only teams that appear in the JSON ``instrTeams_instrTPG.Teams`` array
    (non-zero measured latency) are added to the TPG.  Zero-latency teams
    are parsed from the disassembly but not added, since they carry no
    training signal.

    Args:
        disasm_path: Path to the objdump disassembly file.
        json_path:   Path to the JSON results file.

    Returns:
        ``(tpg, uarch)`` — the populated TPG and the Uarch it was measured on.
    """
    lat_data: TPGLatencyData = Disassembler.parse_latency_json(json_path)

    uarch = Uarch(
        name=lat_data.simulator,
        isa=lat_data.isa,
        abi=lat_data.abi,
    )
    tpg = TPG(
        name=f"tpg_{lat_data.simulator}_{lat_data.dtype}",
        dtype=lat_data.dtype,
    )

    blocks: dict[int, TeamBlock] = Disassembler.parse_tpg_file(disasm_path)

    skipped: list[int] = []
    for team_id, block in sorted(blocks.items()):
        tl = lat_data.get_team_latency(team_id)
        if tl is None:
            skipped.append(team_id)
            continue

        team = build_team_from_block(
            block=block,
            uarch=uarch,
            nb_measurements=tl.nb_measurements,
            avg_latency=tl.avg_cycles,
            stddev=tl.stddev_cycles,
            cv=tl.coefficient_variation,
        )
        tpg.add_team(team)

    print(
        f"[build_tpg] Parsed {len(blocks)} teams from disassembly. "
        f"Added {len(tpg.teams)} with measured latency, "
        f"skipped {len(skipped)} zero-latency teams: {skipped}"
    )
    return tpg, uarch


def build_synthetic_team(
    team_id: int,
    asm_lines: list[str],
    uarch: Uarch,
    latency: float,
) -> Team:
    """Build a Team from a list of raw assembly strings (no file needed)."""
    instructions = [Instruction.parse(line) for line in asm_lines]
    code = "\n".join(asm_lines)

    team = Team(id=team_id, code=code)
    team.set_instructions(instructions)

    fv = FeaturesAnalyzer.analyze_instructions(instructions, id_team=team_id)
    team.set_feature_vector(fv)

    # Synthetic teams have a single "run" by convention
    meas = TeamMeasurement(
        uarch=uarch,
        id_team=team_id,
        latency=latency,
        nb_measurements=1,
    )
    team.add_measurement(meas)
    uarch.add_measurement(meas)
    return team


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) == 3:
        disasm_path = sys.argv[1]
        json_path   = sys.argv[2]
        print(f"[main] Disassembly : {disasm_path}")
        print(f"[main] Latency JSON: {json_path}")
        tpg, uarch = build_tpg_from_files(disasm_path, json_path)

    else:
        print("[main] No files given — using synthetic assembly.")
        uarch = Uarch(name="CV32E40X", isa="RV32IMCZba_Zbb", abi="ilp32")
        tpg   = TPG(name="tpg_synthetic", dtype="int32", gpis=["gpi_0", "gpi_1"])

        synthetic_blocks = [
            (["addi a5,a5,4", "lw a4,0(a5)", "add a3,a4,a5", "sw a3,0(a5)"],   10.5),
            (["mul a0,a1,a2", "add a3,a0,a4", "sw a3,8(sp)", "lw a5,-4(sp)"],   12.0),
            (["div a0,a1,a2", "rem a3,a4,a5", "add a0,a0,a3"],                   72.0),
            (["beq a0,a1,8", "addi a0,a0,1", "jal ra,0"],                         5.0),
            (["lw a0,0(sp)", "lw a1,4(sp)", "add a2,a0,a1", "sw a2,8(sp)"],      8.0),
        ]
        for tid, (asm_lines, lat) in enumerate(synthetic_blocks):
            tpg.add_team(build_synthetic_team(tid, asm_lines, uarch, lat))

    # -------------------------------------------------------------------
    # Dataset overview — easy inspection of all teams and their latencies
    # -------------------------------------------------------------------
    print_dataset_summary(tpg)

    # -------------------------------------------------------------------
    # Detailed view of the first team (code + feature vector)
    # -------------------------------------------------------------------
    if tpg.teams:
        t0 = tpg.teams[0]
        print(f"\n[Team {t0.id}] Raw disassembly (Team.code):")
        print(t0.code)
        for instr in t0.instructions:
            print(instr)
        if t0.feature_vector:
            print(f"\n[Team {t0.id}] FeatureVector (first 10 features):")
            for k, v in list(t0.feature_vector.values.items())[:10]:
                print(f"  {k}: {v}")

    # -------------------------------------------------------------------
    # Regression
    # -------------------------------------------------------------------
    teams_with_data = [
        t for t in tpg.teams
        if t.feature_vector and t.mean_latency() is not None
    ]

    if len(teams_with_data) >= 2:
        feature_vectors = [t.feature_vector for t in teams_with_data]
        latencies       = [t.mean_latency() for t in teams_with_data]  # type: ignore[misc]

        print(f"\n[Regressor] Training on {len(feature_vectors)} samples …")
        model = Regressor.train(feature_vectors, latencies)
        print(f"[Regressor] {model}")

        first_fv   = teams_with_data[0].feature_vector
        prediction = Regressor.predict(model, first_fv)    # type: ignore[arg-type]
        actual     = teams_with_data[0].mean_latency()
        print(
            f"\n[Predict] team_id={teams_with_data[0].id}  "
            f"actual={actual:.2f}  predicted={prediction:.2f}"
        )
    else:
        print("\n[Regressor] Need at least 2 teams to train — skipping.")


if __name__ == "__main__":
    main()