"""Atom and bond type definitions for molecule templates."""

from dataclasses import dataclass, field
from enum import Enum, auto


class AtomCategory(Enum):
    """Categories of atoms based on element and hybridization."""

    C_SP3 = auto()
    C_SP2 = auto()
    C_SP2_AROM = auto()
    C_SP = auto()
    N = auto()
    N_AROM = auto()
    N_POS = auto()
    O = auto()
    O_NEG = auto()
    S = auto()
    P = auto()
    F = auto()
    CL = auto()
    BR = auto()
    I = auto()
    H = auto()
    OTHER = auto()


class BondCategory(Enum):
    """Categories of bonds."""

    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()
    AROMATIC = auto()


@dataclass
class AtomType:
    """Atom type with element and properties.

    Attributes:
        category: The atom category.
        label: Human-readable label.
        vdw_radius: Van der Waals radius in Angstroms.
        color: Default RGB color for visualization.
    """

    category: AtomCategory
    label: str
    vdw_radius: float
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __hash__(self) -> int:
        return hash((self.category, self.label))


@dataclass
class BondType:
    """Bond type with properties.

    Attributes:
        category: The bond category.
        label: Human-readable label.
        radius: Bond cylinder radius for visualization.
        color: Default RGB color.
    """

    category: BondCategory
    label: str
    radius: float = 0.1
    color: tuple[float, float, float] = (0.3, 0.3, 0.3)

    def __hash__(self) -> int:
        return hash((self.category, self.label))


# Van der Waals radii in Angstroms (from RDKit/literature)
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

# CPK-style colors for atoms
ATOM_COLORS: dict[str, tuple[float, float, float]] = {
    "H": (1.0, 1.0, 1.0),
    "C": (0.2, 0.2, 0.2),
    "N": (0.0, 0.0, 1.0),
    "O": (1.0, 0.0, 0.0),
    "F": (0.0, 1.0, 0.0),
    "P": (1.0, 0.5, 0.0),
    "S": (1.0, 1.0, 0.0),
    "Cl": (0.0, 1.0, 0.0),
    "Br": (0.6, 0.1, 0.1),
    "I": (0.4, 0.0, 0.7),
}


@dataclass
class TypeVocabulary:
    """Vocabulary mapping type IDs to atom/bond types.

    Attributes:
        atom_types: List of atom types (index = type_id for atoms).
        bond_types: List of bond types (index = type_id for bonds).
        atom_to_id: Mapping from AtomCategory to type_id.
        bond_to_id: Mapping from BondCategory to type_id.
    """

    atom_types: list[AtomType] = field(default_factory=list)
    bond_types: list[BondType] = field(default_factory=list)
    atom_to_id: dict[AtomCategory, int] = field(default_factory=dict)
    bond_to_id: dict[BondCategory, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Build reverse mappings."""
        self.atom_to_id = {at.category: i for i, at in enumerate(self.atom_types)}
        self.bond_to_id = {
            bt.category: i + len(self.atom_types)
            for i, bt in enumerate(self.bond_types)
        }

    @property
    def num_types(self) -> int:
        """Total number of types (atoms + bonds)."""
        return len(self.atom_types) + len(self.bond_types)

    @property
    def num_atom_types(self) -> int:
        """Number of atom types."""
        return len(self.atom_types)

    @property
    def num_bond_types(self) -> int:
        """Number of bond types."""
        return len(self.bond_types)

    def get_type_id(self, category: AtomCategory | BondCategory) -> int:
        """Get type ID for an atom or bond category.

        Args:
            category: AtomCategory or BondCategory.

        Returns:
            Integer type ID.
        """
        if isinstance(category, AtomCategory):
            return self.atom_to_id[category]
        return self.bond_to_id[category]

    def get_label(self, type_id: int) -> str:
        """Get label for a type ID.

        Args:
            type_id: Integer type ID.

        Returns:
            Human-readable label.
        """
        if type_id < len(self.atom_types):
            return self.atom_types[type_id].label
        bond_idx = type_id - len(self.atom_types)
        return self.bond_types[bond_idx].label

    def is_bond(self, type_id: int) -> bool:
        """Check if type ID corresponds to a bond."""
        return type_id >= len(self.atom_types)

    def get_atom_colors(self) -> list[tuple[float, float, float]]:
        """Get default colors for all atom types.

        Returns:
            List of RGB tuples for each atom type.
        """
        return [at.color for at in self.atom_types]

    @classmethod
    def default(cls, include_bonds: bool = True) -> "TypeVocabulary":
        """Create default vocabulary with common atom and bond types.

        Args:
            include_bonds: Whether to include bond types.

        Returns:
            TypeVocabulary with default types.
        """
        atom_types = [
            AtomType(
                AtomCategory.C_SP3,
                "C_sp3",
                VDW_RADII["C"],
                ATOM_COLORS["C"],
            ),
            AtomType(
                AtomCategory.C_SP2,
                "C_sp2",
                VDW_RADII["C"],
                ATOM_COLORS["C"],
            ),
            AtomType(
                AtomCategory.C_SP2_AROM,
                "C_sp2_arom",
                VDW_RADII["C"],
                ATOM_COLORS["C"],
            ),
            AtomType(
                AtomCategory.N,
                "N",
                VDW_RADII["N"],
                ATOM_COLORS["N"],
            ),
            AtomType(
                AtomCategory.N_AROM,
                "N_arom",
                VDW_RADII["N"],
                ATOM_COLORS["N"],
            ),
            AtomType(
                AtomCategory.O,
                "O",
                VDW_RADII["O"],
                ATOM_COLORS["O"],
            ),
            AtomType(
                AtomCategory.S,
                "S",
                VDW_RADII["S"],
                ATOM_COLORS["S"],
            ),
            AtomType(
                AtomCategory.F,
                "F",
                VDW_RADII["F"],
                ATOM_COLORS["F"],
            ),
            AtomType(
                AtomCategory.CL,
                "Cl",
                VDW_RADII["Cl"],
                ATOM_COLORS["Cl"],
            ),
            AtomType(
                AtomCategory.BR,
                "Br",
                VDW_RADII["Br"],
                ATOM_COLORS["Br"],
            ),
            AtomType(
                AtomCategory.H,
                "H",
                VDW_RADII["H"],
                ATOM_COLORS["H"],
            ),
            AtomType(
                AtomCategory.OTHER,
                "Other",
                1.7,
                (0.5, 0.5, 0.5),
            ),
        ]

        bond_types = []
        if include_bonds:
            bond_types = [
                BondType(BondCategory.SINGLE, "single", 0.08),
                BondType(BondCategory.DOUBLE, "double", 0.10),
                BondType(BondCategory.TRIPLE, "triple", 0.12),
                BondType(BondCategory.AROMATIC, "aromatic", 0.09),
            ]

        return cls(atom_types=atom_types, bond_types=bond_types)
