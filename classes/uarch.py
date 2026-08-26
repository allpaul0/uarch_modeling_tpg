from __future__ import annotations

from dataclasses import dataclass

from .isa import ISA


@dataclass
class Uarch:
    """
    Describes a micro-architecture: a simulation target that *implements* an
    ISA and on which CompiledTeams are executed and measured.

    v2 change
    ---------
    ``isa`` is now an :class:`ISA` object shared with every CompiledTeam
    compiled for that ISA (``Uarch "1..*" --> "1" ISA : implements``), instead
    of a bare string duplicated on each Uarch.  Use :attr:`isa_name` where the
    plain string is wanted.

    Attributes
    ----------
    name: Simulator identifier (e.g. "cv32e40x_im2_zba_zbb").
    isa:  The ISA this micro-architecture implements.
    abi:  ABI string used during compilation (e.g. "ilp32").
    """
    name: str
    isa: ISA
    abi: str = ""

    @property
    def isa_name(self) -> str:
        """The ISA string (convenience accessor, matches the v1 attribute)."""
        return self.isa.name if self.isa is not None else ""

    def implements(self, isa: "ISA | str") -> bool:
        """True if this uarch implements *isa* (ISA object or ISA name)."""
        name = isa.name if isinstance(isa, ISA) else isa
        return self.isa_name == name

    # ------------------------------------------------------------------ #
    # Backwards compatibility with v1 pickles
    # ------------------------------------------------------------------ #

    def __setstate__(self, state: dict) -> None:
        # v1 stored `isa` as a plain string; wrap it in an ISA object.
        isa = state.get("isa")
        if isinstance(isa, str):
            state["isa"] = ISA(name=isa)
        elif isa is None:
            state["isa"] = ISA(name="")
        state.setdefault("abi", "")
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (f"Uarch(name={self.name!r}, isa={self.isa_name!r}, "
                f"abi={self.abi!r})")