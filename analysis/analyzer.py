from __future__ import annotations
from collections import Counter

from classes.instruction import Instruction
from classes.features import FeatureVector

# ---------------------------------------------------------------------------
# Instruction categories — kept for use in RAW hazard / register analysis
# only.  They are NOT used to produce category-level count features because
# those would be linear combinations of the per-mnemonic counts.
# ---------------------------------------------------------------------------

_MEM_STORES = {"sw", "sh", "sb"}
_BRANCHES   = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
_JUMPS      = {"jal", "jalr"}


class FeaturesAnalyzer:
    """
    Extracts a FeatureVector from a list of Instructions.

    Features produced
    -----------------
    Per-mnemonic counts  → "<MNEM>_count"
        One entry per distinct mnemonic observed.  These are the atomic,
        independent count features.  Category-level aggregates (load_count,
        store_count, …) are intentionally NOT included: they are exact linear
        combinations of the per-mnemonic counts (e.g. load_count = lw_count +
        lh_count + …) and would introduce perfect multicollinearity, corrupting
        any linear model including Lasso.  total_instructions is likewise the
        sum of all per-mnemonic counts and is excluded for the same reason.

    RAW hazard estimate  → "raw_hazards"
        Count of Read-After-Write hazards within a 2-instruction look-back
        window.  This is a derived structural feature not expressible as a
        linear combination of mnemonic counts, so it is kept.

    Bigram transitions   → "<A>_<B>_transition"
        Counts of consecutive mnemonic pairs.  These capture instruction-
        ordering information that individual counts cannot express.
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
        # Each mnemonic gets its own independent count feature.
        # No category aggregates and no total_instructions — they are linear
        # combinations of these counts and must not appear alongside them.
        mnem_counts = Counter(mnemonics)
        for mnem, cnt in mnem_counts.items():
            values[f"{mnem}_count"] = float(cnt)

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