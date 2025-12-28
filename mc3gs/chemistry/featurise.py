"""Feature extraction from RDKit molecules."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from mc3gs.chemistry.typing import AtomCategory, BondCategory, TypeVocabulary


def get_atom_category(atom: Chem.Atom) -> AtomCategory:
    """Determine atom category from RDKit atom.

    Args:
        atom: RDKit Atom object.

    Returns:
        AtomCategory for the atom.
    """
    symbol = atom.GetSymbol()
    is_aromatic = atom.GetIsAromatic()
    hybridization = atom.GetHybridization()
    formal_charge = atom.GetFormalCharge()

    if symbol == "C":
        if is_aromatic:
            return AtomCategory.C_SP2_AROM
        if hybridization == Chem.HybridizationType.SP3:
            return AtomCategory.C_SP3
        if hybridization == Chem.HybridizationType.SP2:
            return AtomCategory.C_SP2
        if hybridization == Chem.HybridizationType.SP:
            return AtomCategory.C_SP
        return AtomCategory.C_SP3
    elif symbol == "N":
        if is_aromatic:
            return AtomCategory.N_AROM
        if formal_charge > 0:
            return AtomCategory.N_POS
        return AtomCategory.N
    elif symbol == "O":
        if formal_charge < 0:
            return AtomCategory.O_NEG
        return AtomCategory.O
    elif symbol == "S":
        return AtomCategory.S
    elif symbol == "P":
        return AtomCategory.P
    elif symbol == "F":
        return AtomCategory.F
    elif symbol == "Cl":
        return AtomCategory.CL
    elif symbol == "Br":
        return AtomCategory.BR
    elif symbol == "I":
        return AtomCategory.I
    elif symbol == "H":
        return AtomCategory.H
    else:
        return AtomCategory.OTHER


def get_bond_category(bond: Chem.Bond) -> BondCategory:
    """Determine bond category from RDKit bond.

    Args:
        bond: RDKit Bond object.

    Returns:
        BondCategory for the bond.
    """
    bond_type = bond.GetBondType()

    if bond.GetIsAromatic():
        return BondCategory.AROMATIC
    elif bond_type == Chem.BondType.SINGLE:
        return BondCategory.SINGLE
    elif bond_type == Chem.BondType.DOUBLE:
        return BondCategory.DOUBLE
    elif bond_type == Chem.BondType.TRIPLE:
        return BondCategory.TRIPLE
    else:
        return BondCategory.SINGLE


def featurise_molecule(
    mol: Chem.Mol,
    vocab: TypeVocabulary,
    include_hydrogens: bool = False,
    include_bonds: bool = True,
    conformer_id: int = 0,
) -> dict:
    """Extract Gaussian splat features from a molecule.

    Args:
        mol: RDKit molecule with 3D conformer.
        vocab: Type vocabulary for mapping categories to IDs.
        include_hydrogens: Whether to include hydrogen atoms.
        include_bonds: Whether to include bond Gaussians.
        conformer_id: Which conformer to use.

    Returns:
        Dictionary with:
            - positions: np.ndarray [N, 3] - Gaussian centers
            - covariances: np.ndarray [N, 3, 3] - Local covariance matrices
            - type_ids: np.ndarray [N] - Type IDs
            - labels: list[str] - Type labels
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers. Generate 3D coordinates first.")

    conformer = mol.GetConformer(conformer_id)

    positions = []
    covariances = []
    type_ids = []
    labels = []

    # Process atoms
    for atom in mol.GetAtoms():
        if not include_hydrogens and atom.GetSymbol() == "H":
            continue

        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)
        pos_array = np.array([pos.x, pos.y, pos.z])

        category = get_atom_category(atom)

        # Get VdW radius for covariance
        try:
            type_id = vocab.get_type_id(category)
            atom_type = vocab.atom_types[type_id]
            radius = atom_type.vdw_radius
        except (KeyError, IndexError):
            # Fallback to OTHER
            type_id = vocab.get_type_id(AtomCategory.OTHER)
            radius = 1.7

        # Spherical covariance based on VdW radius
        # Scale radius to reasonable Gaussian sigma
        sigma = radius * 0.4  # Empirical scaling
        cov = np.eye(3) * (sigma**2)

        positions.append(pos_array)
        covariances.append(cov)
        type_ids.append(type_id)
        labels.append(vocab.get_label(type_id))

    # Process bonds
    if include_bonds and vocab.num_bond_types > 0:
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Skip bonds to hydrogens if not including them
            if not include_hydrogens:
                if (
                    mol.GetAtomWithIdx(begin_idx).GetSymbol() == "H"
                    or mol.GetAtomWithIdx(end_idx).GetSymbol() == "H"
                ):
                    continue

            begin_pos = conformer.GetAtomPosition(begin_idx)
            end_pos = conformer.GetAtomPosition(end_idx)

            begin_array = np.array([begin_pos.x, begin_pos.y, begin_pos.z])
            end_array = np.array([end_pos.x, end_pos.y, end_pos.z])

            # Bond center
            center = (begin_array + end_array) / 2
            positions.append(center)

            # Bond direction
            direction = end_array - begin_array
            length = np.linalg.norm(direction)
            if length > 1e-6:
                direction = direction / length
            else:
                direction = np.array([1.0, 0.0, 0.0])

            # Anisotropic covariance: elongated along bond axis
            category = get_bond_category(bond)
            try:
                type_id = vocab.get_type_id(category)
                bond_type = vocab.bond_types[type_id - vocab.num_atom_types]
                radius = bond_type.radius
            except (KeyError, IndexError):
                type_id = vocab.get_type_id(BondCategory.SINGLE)
                radius = 0.08

            # Build covariance matrix
            # Elongated along bond axis, narrow perpendicular
            sigma_along = length * 0.4  # Along bond
            sigma_perp = radius  # Perpendicular

            # Create orthonormal basis with direction as first axis
            if abs(direction[0]) < 0.9:
                perp1 = np.cross(direction, np.array([1, 0, 0]))
            else:
                perp1 = np.cross(direction, np.array([0, 1, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)

            # Build rotation matrix (columns are basis vectors)
            R = np.column_stack([direction, perp1, perp2])
            S = np.diag([sigma_along**2, sigma_perp**2, sigma_perp**2])
            cov = R @ S @ R.T

            covariances.append(cov)
            type_ids.append(type_id)
            labels.append(vocab.get_label(type_id))

    return {
        "positions": np.array(positions, dtype=np.float32),
        "covariances": np.array(covariances, dtype=np.float32),
        "type_ids": np.array(type_ids, dtype=np.int64),
        "labels": labels,
    }


def compute_molecular_properties(mol: Chem.Mol) -> dict:
    """Compute molecular properties for analysis.

    Args:
        mol: RDKit molecule.

    Returns:
        Dictionary of molecular properties.
    """
    return {
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "molecular_weight": rdMolDescriptors.CalcExactMolWt(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }
