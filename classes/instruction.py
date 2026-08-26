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

# Mnemonics whose immediate/target operand identifies a *callee*, not just a
# displacement.  For these the symbol objdump prints after the address
# ("jal ra,884 <expf>") is semantically load-bearing: "jal expf" and
# "jal logf" are different operations with wildly different latencies, so the
# callee is kept as part of the instruction's identity (see `operation`).
#
# Plain local jumps (j / jr / c.j) are deliberately excluded: their target is
# intra-block control flow, not a distinct operation.
_CALL_MNEMONICS: frozenset[str] = frozenset({
    "jal", "jalr", "call", "tail",
    "c.jal", "c.jalr",
})

# "<expf>" / "<T2_start+0x1e>"  →  captures the symbol name without the
# "+0x..." offset objdump appends.
_SYMBOL_RE = re.compile(r"<([^>+\s]+)(?:\+0x[0-9a-f]+)?>")

# Characters that would break a feature name built from an operation.
_UNSAFE_SYMBOL_CHARS = re.compile(r"[^A-Za-z0-9_.]+")

# Bumped whenever parsing changes what an Instruction records.
#   1 — call targets discarded (every call collapsed into "jal")
#   2 — call targets kept, so "jal expf" and "jal logf" are distinct
# CompiledTeam stores the version its instructions were produced with and
# re-parses from its stored disassembly text when it is out of date.
INSTRUCTION_PARSER_VERSION = 2


def _clean_symbol(symbol: str | None) -> str | None:
    """
    Normalise an objdump symbol annotation into a bare callee name.

    ``"<expf>"`` → ``"expf"``, ``"<T2_start+0x1e>"`` → ``"T2_start"``,
    ``"expf"`` → ``"expf"``.  Returns None if nothing usable is left.
    """
    if not symbol:
        return None
    m = _SYMBOL_RE.search(symbol)
    name = m.group(1) if m else symbol.strip().strip("<>")
    name = name.split("+", 1)[0].strip()
    name = _UNSAFE_SYMBOL_CHARS.sub("_", name).strip("_")
    return name or None


@dataclass
class Instruction:
    """
    A single disassembled RISC-V instruction belonging to a Team.

    Attributes
    ----------
    mnemonic: The bare mnemonic ("jal", "fadd.s", …).
    operands: Normalised operands; immediates are replaced by "CONST".
    raw:      The original assembly text.
    target:   For call instructions only, the callee symbol resolved by
              objdump ("expf", "logf", …), or None.  Kept out of `operands`
              so register analysis is unaffected.
    """
    mnemonic: str
    operands: list[str]
    raw: str = ""
    target: str | None = None

    # ------------------------------------------------------------------ #
    # Identity used by feature extraction
    # ------------------------------------------------------------------ #

    @property
    def operation(self) -> str:
        """
        The operation this instruction performs, as used for feature names.

        Normally the lowercased mnemonic.  For a call with a resolved callee
        the target is folded in — ``jal_expf``, ``jal_logf``, ``jal_sinf`` —
        so that each library call is counted as its own operation instead of
        all of them collapsing into a single ``jal``.  Calls through a
        register, where objdump resolves no symbol, stay plain ``jalr``.
        """
        base = self.mnemonic.lower()
        if self.target:
            return f"{base}_{self.target}"
        return base

    def __setstate__(self, state: dict) -> None:
        # Instructions pickled before call targets were kept have no `target`.
        state.setdefault("target", None)
        self.__dict__.update(state)

    @property
    def is_call(self) -> bool:
        """True if this instruction is a call whose callee may be known."""
        return self.mnemonic.lower() in _CALL_MNEMONICS

    def __repr__(self) -> str:
        target = f", target={self.target!r}" if self.target else ""
        return (
            f"Instruction(mnemonic={self.mnemonic!r}, "
            f"operands={self.operands}{target})"
        )

    @staticmethod
    def parse(asm: str, symbol: str | None = None) -> "Instruction":
        """
        Parse a single RISC-V assembly string such as:
            'sw  a5,-172(s0)'
            'addi a5,a5,4'
            'li   a5,0'
            'jal  ra,884 <expf>'

        Args:
            asm:    The instruction text.  A trailing "<symbol>" annotation is
                    accepted here and used as the call target when *symbol* is
                    not supplied.
            symbol: The symbol annotation objdump printed for this line, when
                    the caller stripped it before calling (as the disassembler
                    line regex does).  Only used for call mnemonics.
        """
        asm = asm.strip()
        parts = asm.split(None, 1)
        mnemonic = parts[0]
        operands_part = parts[1] if len(parts) > 1 else ""

        raw_operands = [op.strip() for op in operands_part.split(",") if op.strip()]

        operands: list[str] = []
        inline_symbol: str | None = None

        for op in raw_operands:
            # Strip the symbolic annotation that objdump appends to branch/jump
            # targets, e.g. "886 <T2_start+0x1e>" → "886".
            # These are NOT #-prefixed comments (those are already removed by
            # _LINE_RE in the disassembler); they are bare "<label+offset>"
            # suffixes separated from the numeric address by a space.
            # The annotation is remembered: for a call it names the callee.
            m_sym = re.search(r"\s+(<[^>]+>)$", op)
            if m_sym:
                inline_symbol = m_sym.group(1)
                op = op[: m_sym.start()]

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

        # The callee only matters for calls; for every other instruction the
        # annotation is a plain displacement label and is discarded, exactly
        # as before.
        target: str | None = None
        if mnemonic.lower() in _CALL_MNEMONICS:
            target = _clean_symbol(symbol if symbol is not None else inline_symbol)

        return Instruction(
            mnemonic=mnemonic,
            operands=operands,
            raw=asm,
            target=target,
        )