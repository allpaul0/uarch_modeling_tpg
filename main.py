"""
main.py — end-to-end demonstration of the TPG latency-estimation pipeline.

Usage
-----
    python main.py                        # uses synthetic data
    python main.py path/to/dump.txt       # parse a real objdump file
"""

from __future__ import annotations
import sys

from classes import TPG, Team, Instruction, Uarch, TeamMeasurement, FeatureVector
from analysis import Disassembler, FeaturesAnalyzer, Regressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_team_from_file(team_id: int, path: str, uarch: Uarch) -> Team:
    """Parse a dump file, extract features, and attach a synthetic measurement."""
    instructions = Disassembler.parse_file(path)
    code_snippet = f"<loaded from {path}>"

    team = Team(id=team_id, code=code_snippet)
    team.set_instructions(instructions)

    fv = FeaturesAnalyzer.analyze_instructions(instructions, id_team=team_id)
    team.set_feature_vector(fv)

    # Estimated latency as a proxy for a real measurement
    estimated = float(Disassembler.estimate_block_latency(instructions))
    meas = TeamMeasurement(uarch=uarch, id_team=team_id, latency=estimated)
    team.add_measurement(meas)
    uarch.add_measurement(meas)

    return team


def build_synthetic_team(team_id: int, asm_lines: list[str], uarch: Uarch, latency: float) -> Team:
    """Build a Team from a list of raw assembly strings (no file needed)."""
    code = "\n".join(asm_lines)
    instructions = [instr for line in asm_lines for instr in [Instruction.parse(line)]]

    team = Team(id=team_id, code=code)
    team.set_instructions(instructions)

    fv = FeaturesAnalyzer.analyze_instructions(instructions, id_team=team_id)
    team.set_feature_vector(fv)

    meas = TeamMeasurement(uarch=uarch, id_team=team_id, latency=latency)
    team.add_measurement(meas)
    uarch.add_measurement(meas)

    return team


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Define micro-architecture
    uarch = Uarch(name="CV32E40X", isa="RV32IMCZba_Zbb_Zbc_Zbs", abi="ilp32")

    # 2. Create TPG container
    tpg = TPG(name="tpg_inference_fixedpt", dtype="int32", gpis=["gpi_0", "gpi_1"])

    # 3. Build teams
    if len(sys.argv) > 1:
        # Real file path provided
        dump_path = sys.argv[1]
        print(f"[main] Parsing dump file: {dump_path}")
        team = build_team_from_file(team_id=0, path=dump_path, uarch=uarch)
        tpg.add_team(team)
    else:
        # Synthetic teams for demonstration
        print("[main] No dump file given — using synthetic assembly.")
        synthetic_blocks = [
            (["addi a5,a5,4", "lw a4,0(a5)", "add a3,a4,a5", "sw a3,0(a5)"], 10.5),
            (["mul a0,a1,a2", "add a3,a0,a4", "sw a3,8(sp)", "lw a5,-4(sp)"], 12.0),
            (["div a0,a1,a2", "rem a3,a4,a5", "add a0,a0,a3"], 72.0),
            (["beq a0,a1,8", "addi a0,a0,1", "jal ra,0"], 5.0),
            (["lw a0,0(sp)", "lw a1,4(sp)", "add a2,a0,a1", "sw a2,8(sp)"], 8.0),
        ]
        for tid, (asm_lines, lat) in enumerate(synthetic_blocks):
            team = build_synthetic_team(tid, asm_lines, uarch, lat)
            tpg.add_team(team)

    print(f"\n[TPG] {tpg}")
    for team in tpg.teams:
        print(f"  {team}")
        if team.feature_vector:
            print(f"    FeatureVector (first 8): "
                  f"{ {k: v for k, v in list(team.feature_vector.values.items())[:8]} }")
    
    # print instructions for the first team
    if tpg.teams and tpg.teams[0].instructions:
        print(f"\n[Instructions] Team {tpg.teams[0].id} instructions:")
        for instr in tpg.teams[0].instructions:
            print(f"  {instr}")

    # 4. Train a regression model (requires ≥2 teams with measurements)
    teams_with_data = [t for t in tpg.teams if t.feature_vector and t.mean_latency() is not None]

    if len(teams_with_data) >= 2:
        feature_vectors = [t.feature_vector for t in teams_with_data]
        latencies       = [t.mean_latency() for t in teams_with_data]  # type: ignore[misc]

        print(f"\n[Regressor] Training on {len(feature_vectors)} samples …")
        model = Regressor.train(feature_vectors, latencies)
        print(f"[Regressor] {model}")

        # 5. Predict on the first team
        first_fv = teams_with_data[0].feature_vector
        prediction = Regressor.predict(model, first_fv)  # type: ignore[arg-type]
        actual     = teams_with_data[0].mean_latency()
        print(f"\n[Predict] team_id={teams_with_data[0].id}  "
              f"actual={actual:.2f}  predicted={prediction:.2f}")
    else:
        print("\n[Regressor] Need at least 2 teams to train — skipping.")


if __name__ == "__main__":
    main()