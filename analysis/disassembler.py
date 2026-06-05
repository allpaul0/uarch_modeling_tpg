from __future__ import annotations
import re

from classes.team import Instruction


# ---------------------------------------------------------------------------
# Default / table-based latencies (CV32E40X)
# ---------------------------------------------------------------------------

INTEGER_LATENCY = 1  # default for any RV32I integer instruction

INSTRUCTION_LATENCY: dict[str, int] = {
    # Memory
    "lw": 1, "lh": 1, "lb": 1, "lhu": 1, "lbu": 1,
    "sw": 1, "sh": 1, "sb": 1,
    # Control flow
    "jal": 2, "jalr": 2,
    # Branches
    "beq": 1, "bne": 1, "blt": 1, "bge": 1, "bltu": 1, "bgeu": 1,
    # Multiply
    "mul": 1, "mulh": 4, "mulhu": 4, "mulhsu": 4,
    # Division
    "div": 35, "divu": 35, "rem": 35, "remu": 35,
    # CSR
    "csrrw": 1, "csrrs": 1, "csrrc": 1,
    "csrrwi": 1, "csrrsi": 1, "csrrci": 1,
    # System / fence
    "mret": 2, "fence": 5, "fence.i": 5,
    # Bit-manip (Zba/Zbb/Zbc/Zbs)
    "clz": 1, "ctz": 1, "pcnt": 1, "max": 1, "min": 1,
}

# Regex matching one disassembled line:
#   <addr>:  <machine-code>   <mnemonic> [operands]   [# optional comment]
_LINE_RE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(.+?)(?:\s*#.*)?$")


class Disassembler:
    """
    Parses raw RISC-V assembly text (e.g. output of objdump -d) into
    a list of Instruction objects.

    All methods are static — no instance state needed.
    """

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #


    @staticmethod
    def parse_assembly(code: str) -> list[Instruction]:
        """
        Parse a multi-line assembly string and return one Instruction per
        decoded line.  Comment-only lines and blank lines are silently skipped.

        Args:
            code: Raw text from objdump (or similar disassembler).

        Returns:
            Ordered list of Instruction objects.
        """
        instructions: list[Instruction] = []
        for line in code.splitlines():
            m = _LINE_RE.search(line)
            if m:
                instructions.append(Instruction.parse(m.group(1).strip()))
        return instructions

    @staticmethod
    def parse_file(path: str) -> list[Instruction]:
        """
        Convenience wrapper: read a file and call parse_assembly.

        Args:
            path: Path to the objdump output file.

        Returns:
            Ordered list of Instruction objects.
        """
        with open(path, "r") as fh:
            return Disassembler.parse_assembly(fh.read())

    # ------------------------------------------------------------------ #
    # Latency helpers (kept here because they depend on the parsed result)
    # ------------------------------------------------------------------ #

    @staticmethod
    def instruction_latency(instr: Instruction) -> int:
        """Return the estimated cycle latency for a single instruction."""
        lat = INSTRUCTION_LATENCY.get(instr.mnemonic.lower())
        if lat is None:
            print(
                f"Warning: unknown mnemonic '{instr.mnemonic}', "
                f"using default latency {INTEGER_LATENCY}"
            )
            return INTEGER_LATENCY
        return lat

    @staticmethod
    def estimate_block_latency(instructions: list[Instruction]) -> int:
        """
        Naïve basic-block latency estimate: sum of per-instruction latencies.

        Args:
            instructions: Ordered list of Instructions in the block.

        Returns:
            Total estimated cycles.
        """
        return sum(Disassembler.instruction_latency(i) for i in instructions)