"""
classes/isa.py — the Instruction-Set Architecture a CompiledTeam targets.

Introduced by the v2 model.  In v1 the ISA string was an attribute of Uarch
and a CompiledTeam pointed straight at a Uarch.  v2 separates the two levels:

    CompiledTeam  --compiled for-->  ISA  <--implements--  Uarch

The ISA determines the *code* (and therefore the instructions and the static
feature vector); the Uarch determines the *timing* (and therefore the
measurements).  One CompiledTeam can consequently be measured on every Uarch
that implements its ISA.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ISA:
    """
    An instruction-set architecture used as a compilation target.

    Attributes
    ----------
    name: ISA string as reported by ``latencies.json`` and used on the
          compiler command line (e.g. "rv32ic_zicsr_zmmul_zba_zbb").
    """
    name: str

    def __repr__(self) -> str:
        return f"ISA(name={self.name!r})"