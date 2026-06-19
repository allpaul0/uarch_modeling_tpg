from dataclasses import dataclass


@dataclass
class Uarch:
    """
    Describes a micro-architecture compilation and simulation target.

    Attributes
    ----------
    name: Simulator identifier (e.g. "cv32e40x_im2_zba_zbb").
    isa:  ISA string used during compilation (e.g. "rv32im_zicsr_zba_zbb").
    abi:  ABI string used during compilation (e.g. "ilp32").
    """
    name: str
    isa: str
    abi: str

    def __repr__(self) -> str:
        return f"Uarch(name={self.name!r}, isa={self.isa!r}, abi={self.abi!r})"