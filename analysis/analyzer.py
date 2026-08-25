from __future__ import annotations
import re
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

# Maximum look-back distance considered when searching for RAW hazards.
_RAW_MAX_DISTANCE = 1#3


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

    RAW hazards          → "raw_d<N>_<PRODUCER>_<CONSUMER>_count"
        Count of Read-After-Write hazards, broken down by:
          - look-back distance N (1, 2 or 3 instructions apart), and
          - the (producer mnemonic, consumer mnemonic) pair involved.
        A distance-1 hazard between e.g. "lw" and "add" is tracked
        separately from a distance-1 hazard between "lw" and "sub", and
        separately again from a distance-2 hazard between the same two
        mnemonics — these correspond to different pipeline stall/forwarding
        situations and are not interchangeable signals.  No aggregate
        "raw_hazards" total is emitted, for the same multicollinearity
        reason per-mnemonic counts replace total_instructions: the total
        would be an exact sum of these finer-grained features.

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
    ) -> FeatureVector:
        """
        Analyse a list of Instructions and return a FeatureVector.

        Args:
            instructions: Instructions belonging to one Team.

        Returns:
            A populated FeatureVector.
        """
        if not instructions:
            return FeatureVector(values={})

        values: dict[str, float] = {}

        mnemonics = [i.mnemonic.lower() for i in instructions]

        # -- Per-mnemonic counts ------------------------------------------
        # Each mnemonic gets its own independent count feature.
        # No category aggregates and no total_instructions — they are linear
        # combinations of these counts and must not appear alongside them.
        mnem_counts = Counter(mnemonics)
        for mnem, cnt in mnem_counts.items():
            values[f"{mnem}_count"] = float(cnt)

        # -- RAW hazards, split by distance and producer/consumer pair ----
        raw_counts = FeaturesAnalyzer._raw_hazard_counts(instructions)
        for (distance, producer, consumer), cnt in raw_counts.items():
            values[f"raw_d{distance}_{producer}_{consumer}_count"] = float(cnt)

        # -- Bigram transitions -------------------------------------------
        #for (a, b), cnt in FeaturesAnalyzer._bigram_counts(mnemonics).items():
        #    values[f"{a}_{b}_transition"] = float(cnt)

        # -- Length of instructions ---------------------------------------
        #values["BB_length"] = len(instructions)

        return FeatureVector(values=values)

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
        used: set[str] = set()
        for op in instr.operands[1:]:          # operands[0] is typically dest
            m = re.fullmatch(r"-?\d+\((\w+)\)", op)
            if m:
                used.add(m.group(1))
            elif re.fullmatch(r"[a-z][a-z0-9]*", op):
                used.add(op)
        return used

    @staticmethod
    def _raw_hazard_counts(
        instructions: list[Instruction],
        max_distance: int = _RAW_MAX_DISTANCE,
    ) -> Counter:
        """
        Count Read-After-Write (RAW) hazards, keyed by
        (distance, producer_mnemonic, consumer_mnemonic).

        For every instruction (the "consumer") we look back up to
        `max_distance` predecessors. For each look-back distance d in
        1..max_distance, the predecessor at that exact distance (the
        "producer") is checked independently: if it writes a register the
        consumer reads, that's one hazard recorded under
        (d, producer.mnemonic, consumer.mnemonic).

        Distances are evaluated independently of one another (no "stop at
        the first hit" short-circuit), because a distance-1 hazard and a
        distance-2 hazard for the same consumer are different signals — a
        non-forwarding pipeline stalls differently depending on how far
        back the conflicting write happened, and the identity of the
        producer/consumer pair changes which forwarding path would apply.
        At most one hazard is still recorded per (distance, consumer)
        pair: if a producer at that exact distance defines several of the
        registers the consumer reads, that is one structural hazard, not
        several.

        Args:
            instructions: Instructions belonging to one basic block / team.
            max_distance: Largest look-back distance to consider (default 3).

        Returns:
            Counter mapping (distance, producer_mnemonic, consumer_mnemonic)
            to the number of times that exact hazard pattern occurred.
        """
        counts: Counter = Counter()

        for idx in range(1, len(instructions)):
            consumer = instructions[idx]
            uses = FeaturesAnalyzer._used_registers(consumer)
            if not uses:
                continue

            for distance in range(1, max_distance + 1):
                prev_idx = idx - distance
                if prev_idx < 0:
                    break

                producer = instructions[prev_idx]
                defs = FeaturesAnalyzer._defined_registers(producer)
                if uses & defs:
                    key = (
                        distance,
                        producer.mnemonic.lower(),
                        consumer.mnemonic.lower(),
                    )
                    counts[key] += 1

        return counts

    @staticmethod
    def _bigram_counts(mnemonics: list[str]) -> Counter:
        """Return counts of consecutive mnemonic pairs."""
        return Counter(zip(mnemonics, mnemonics[1:]))