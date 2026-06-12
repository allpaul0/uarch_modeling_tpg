from __future__ import annotations
from collections import Counter

from classes.instruction import Instruction
from classes.features import FeatureVector

# ---------------------------------------------------------------------------
# Instruction categories used for feature extraction
# ---------------------------------------------------------------------------

_MEM_LOADS  = {"lw", "lh", "lb", "lhu", "lbu"}
_MEM_STORES = {"sw", "sh", "sb"}
_BRANCHES   = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
_JUMPS      = {"jal", "jalr"}
_MUL        = {"mul", "mulh", "mulhu", "mulhsu"}
_DIV        = {"div", "divu", "rem", "remu"}
_CSR        = {"csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci"}


class FeaturesAnalyzer:
    """
    Extracts a FeatureVector from a list of Instructions.

    Features produced
    -----------------
    Per-mnemonic counts   → "<MNEM>_count"
    Category counts       → "load_count", "store_count", "branch_count",
                             "jump_count", "mul_count", "div_count",
                             "csr_count", "other_count"
    RAW hazard estimate   → "raw_hazards"
    Bigram transitions    → "<A>_<B>_transition"
    Total instructions    → "total_instructions"
    """

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def analyze_instructions(
        instructions: list[Instruction],
        id_team: int = 0,
    ) -> FeatureVector:
        """
        Analyse a list of Instructions and return a FeatureVector.

        Args:
            instructions: Instructions belonging to one Team.
            id_team:      Team identifier forwarded to the FeatureVector.

        Returns:
            A populated FeatureVector.
        """
        if not instructions:
            return FeatureVector(id_team=id_team, values={})

        values: dict[str, float] = {}

        mnemonics = [i.mnemonic.lower() for i in instructions]

        # -- Per-mnemonic counts ------------------------------------------
        mnem_counts = Counter(mnemonics)
        for mnem, cnt in mnem_counts.items():
            values[f"{mnem}_count"] = float(cnt)

        # -- Category counts ----------------------------------------------
        values["load_count"]   = float(sum(mnem_counts[m] for m in _MEM_LOADS  if m in mnem_counts))
        values["store_count"]  = float(sum(mnem_counts[m] for m in _MEM_STORES if m in mnem_counts))
        values["branch_count"] = float(sum(mnem_counts[m] for m in _BRANCHES   if m in mnem_counts))
        values["jump_count"]   = float(sum(mnem_counts[m] for m in _JUMPS      if m in mnem_counts))
        values["mul_count"]    = float(sum(mnem_counts[m] for m in _MUL        if m in mnem_counts))
        values["div_count"]    = float(sum(mnem_counts[m] for m in _DIV        if m in mnem_counts))
        values["csr_count"]    = float(sum(mnem_counts[m] for m in _CSR        if m in mnem_counts))

        known = _MEM_LOADS | _MEM_STORES | _BRANCHES | _JUMPS | _MUL | _DIV | _CSR
        values["other_count"] = float(sum(cnt for m, cnt in mnem_counts.items() if m not in known))

        # -- Total instructions -------------------------------------------
        values["total_instructions"] = float(len(instructions))

        # -- RAW hazard estimate ------------------------------------------
        values["raw_hazards"] = float(FeaturesAnalyzer._count_raw_hazards(instructions))

        # -- Bigram transitions -------------------------------------------
        for (a, b), cnt in FeaturesAnalyzer._bigram_counts(mnemonics).items():
            values[f"{a}_{b}_transition"] = float(cnt)

        return FeatureVector(id_team=id_team, values=values)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _defined_registers(instr: Instruction) -> set[str]:
        """
        Return the set of registers *written* by instr.
        For most RISC-V instructions the destination is operands[0].
        Stores and branches write no register; we exclude them.
        """
        m = instr.mnemonic.lower()
        if m in _MEM_STORES or m in _BRANCHES or m in _JUMPS:
            return set()
        if instr.operands:
            return {instr.operands[0]}
        return set()

    @staticmethod
    def _used_registers(instr: Instruction) -> set[str]:
        """
        Return the set of registers *read* by instr.
        Strips memory-offset syntax like '-8(sp)' → 'sp'.
        """
        import re
        used: set[str] = set()
        for op in instr.operands[1:]:          # operands[0] is typically dest
            m = re.fullmatch(r"-?\d+\((\w+)\)", op)
            if m:
                used.add(m.group(1))
            elif re.fullmatch(r"[a-z][a-z0-9]*", op):
                used.add(op)
        return used

    @staticmethod
    def _count_raw_hazards(instructions: list[Instruction]) -> int:
        """
        Count Read-After-Write (RAW) hazards within a window of 3 instructions.
        A RAW hazard occurs when an instruction reads a register written by
        one of its 2 immediate predecessors.
        """
        hazards = 0
        window = 2  # look-back distance

        for idx in range(1, len(instructions)):
            uses = FeaturesAnalyzer._used_registers(instructions[idx])
            for prev in range(max(0, idx - window), idx):
                defs = FeaturesAnalyzer._defined_registers(instructions[prev])
                if uses & defs:
                    hazards += 1
                    break  # count at most one hazard per instruction
        return hazards

    @staticmethod
    def _bigram_counts(mnemonics: list[str]) -> Counter:
        """Return counts of consecutive mnemonic pairs."""
        return Counter(zip(mnemonics, mnemonics[1:]))