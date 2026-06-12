from dataclasses import dataclass, field
import re

# Complete set of RISC-V register names (ABI names + x0–x31).
# Used by Instruction.parse to distinguish register operands from hex
# branch/jump targets that happen to look like hex strings (e.g. "a5", "a4").
_RISCV_REGISTERS: frozenset[str] = frozenset({
    # Integer — ABI names
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    # Integer — numeric names
    *(f"x{i}" for i in range(32)),
    # Float — ABI names
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "ft8", "ft9", "ft10", "ft11",
    "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
    "fs8", "fs9", "fs10", "fs11",
    "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
    # Float — numeric names
    *(f"f{i}" for i in range(32)),
})


@dataclass
class Instruction:
    """
    A single disassembled RISC-V instruction belonging to a Team.
    """
    mnemonic: str
    operands: list[str]
    raw: str = ""

    def __repr__(self) -> str:
        return (
            f"Instruction(mnemonic={self.mnemonic!r}, "
            f"operands={self.operands})"
        )

    @staticmethod
    def parse(asm: str) -> "Instruction":
        """
        Parse a single RISC-V assembly string such as:
            'sw  a5,-172(s0)'
            'addi a5,a5,4'
            'li   a5,0'
        """
        asm = asm.strip()
        parts = asm.split(None, 1)
        mnemonic = parts[0]
        operands_part = parts[1] if len(parts) > 1 else ""

        raw_operands = [op.strip() for op in operands_part.split(",") if op.strip()]

        operands: list[str] = []

        for op in raw_operands:
            # Strip the symbolic annotation that objdump appends to branch/jump
            # targets, e.g. "886 <T2_start+0x1e>" → "886".
            # These are NOT #-prefixed comments (those are already removed by
            # _LINE_RE in the disassembler); they are bare "<label+offset>"
            # suffixes separated from the numeric address by a space.
            op = re.sub(r"\s+<[^>]+>$", "", op)

            # Case 1: immediate — decimal (e.g. 4, -172) or hex address
            # (e.g. 7da, 24a, bb0) produced by objdump after stripping the
            # label annotation above.
            # We match hex only when the token is NOT a known RISC-V register
            # name, because several registers (a0–a7, t0–t6, s0–s11 …) are
            # valid hex strings and would be misclassified by a pure regex.
            if re.fullmatch(r"-?\d+|0x[0-9a-f]+", op) or (
                re.fullmatch(r"[0-9a-f]+", op) and op not in _RISCV_REGISTERS
            ):
                operands.append("CONST")
                continue

            # Case 2: memory operand like -172(s0)
            m = re.fullmatch(r"(-?\d+)\((\w+)\)", op)
            if m:
                # split into CONST + register
                operands.append("CONST")
                operands.append(m.group(2))
                continue

            # Case 3: normal register (a5, s0, etc.)
            operands.append(op)

        return Instruction(
            mnemonic=mnemonic,
            operands=operands,
            raw=asm,
        )