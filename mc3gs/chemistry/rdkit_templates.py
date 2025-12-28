"""RDKit-based molecule template generation."""

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from mc3gs.chemistry.featurise import featurise_molecule
from mc3gs.chemistry.typing import TypeVocabulary


def generate_conformer(
    mol: Chem.Mol,
    num_conformers: int = 1,
    random_seed: int = 42,
    optimize: bool = True,
    force_field: str = "MMFF",
) -> Chem.Mol:
    """Generate 3D conformer(s) for a molecule.

    Args:
        mol: RDKit molecule (2D or 3D).
        num_conformers: Number of conformers to generate.
        random_seed: Random seed for reproducibility.
        optimize: Whether to optimize geometry with force field.
        force_field: Force field to use ("MMFF" or "UFF").

    Returns:
        Molecule with 3D conformer(s).
    """
    # Work on a copy
    mol = Chem.AddHs(mol)

    # Generate conformers using ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # Use all available threads

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

    if len(conf_ids) == 0:
        # Fallback to simpler embedding
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)

    # Optimize geometry
    if optimize and mol.GetNumConformers() > 0:
        if force_field == "MMFF":
            for conf_id in range(mol.GetNumConformers()):
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                except Exception:
                    # MMFF may fail, try UFF as fallback
                    try:
                        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                    except Exception:
                        pass
        else:
            for conf_id in range(mol.GetNumConformers()):
                try:
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                except Exception:
                    pass

    return mol


def create_template_from_smiles(
    smiles: str,
    vocab: TypeVocabulary | None = None,
    include_hydrogens: bool = False,
    include_bonds: bool = True,
    optimize: bool = True,
    random_seed: int = 42,
) -> dict:
    """Create a molecule template from SMILES string.

    Args:
        smiles: SMILES string representing the molecule.
        vocab: Type vocabulary (uses default if None).
        include_hydrogens: Whether to include hydrogen atoms.
        include_bonds: Whether to include bond Gaussians.
        optimize: Whether to optimize 3D geometry.
        random_seed: Random seed for conformer generation.

    Returns:
        Dictionary with template data:
            - positions: np.ndarray [N, 3]
            - covariances: np.ndarray [N, 3, 3]
            - type_ids: np.ndarray [N]
            - labels: list[str]
            - smiles: str
            - num_atoms: int
            - num_gaussians: int
    """
    if vocab is None:
        vocab = TypeVocabulary.default(include_bonds=include_bonds)

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Generate 3D conformer
    mol = generate_conformer(mol, optimize=optimize, random_seed=random_seed)

    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"Failed to generate conformer for: {smiles}")

    # Extract features
    features = featurise_molecule(
        mol,
        vocab,
        include_hydrogens=include_hydrogens,
        include_bonds=include_bonds,
    )

    # Add metadata
    features["smiles"] = smiles
    features["num_atoms"] = mol.GetNumHeavyAtoms()
    features["num_gaussians"] = len(features["positions"])

    return features


def create_template_from_mol_file(
    path: Path | str,
    vocab: TypeVocabulary | None = None,
    include_hydrogens: bool = False,
    include_bonds: bool = True,
) -> dict:
    """Create a molecule template from a mol/mol2/sdf file.

    Args:
        path: Path to molecule file.
        vocab: Type vocabulary (uses default if None).
        include_hydrogens: Whether to include hydrogen atoms.
        include_bonds: Whether to include bond Gaussians.

    Returns:
        Dictionary with template data.
    """
    path = Path(path)
    if vocab is None:
        vocab = TypeVocabulary.default(include_bonds=include_bonds)

    # Load molecule based on extension
    suffix = path.suffix.lower()
    if suffix in (".mol", ".sdf"):
        mol = Chem.MolFromMolFile(str(path))
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path))
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path))
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if mol is None:
        raise ValueError(f"Failed to load molecule from: {path}")

    # Generate conformer if none exists
    if mol.GetNumConformers() == 0:
        mol = generate_conformer(mol)

    # Extract features
    features = featurise_molecule(
        mol,
        vocab,
        include_hydrogens=include_hydrogens,
        include_bonds=include_bonds,
    )

    features["source_file"] = str(path)
    features["num_atoms"] = mol.GetNumHeavyAtoms()
    features["num_gaussians"] = len(features["positions"])

    return features


def save_template(template: dict, path: Path | str) -> None:
    """Save a molecule template to disk.

    Args:
        template: Template dictionary from create_template_*.
        path: Output path (.npz or .pt).
    """
    path = Path(path)

    if path.suffix == ".npz":
        # Save as numpy
        np.savez(
            path,
            positions=template["positions"],
            covariances=template["covariances"],
            type_ids=template["type_ids"],
            labels=np.array(template["labels"], dtype=object),
            metadata=np.array(
                {
                    k: v
                    for k, v in template.items()
                    if k not in ("positions", "covariances", "type_ids", "labels")
                }
            ),
        )
    elif path.suffix == ".pt":
        # Save as PyTorch
        torch.save(template, path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def load_template(path: Path | str) -> dict:
    """Load a molecule template from disk.

    Args:
        path: Path to template file (.npz or .pt).

    Returns:
        Template dictionary.
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        template = {
            "positions": data["positions"],
            "covariances": data["covariances"],
            "type_ids": data["type_ids"],
            "labels": data["labels"].tolist(),
        }
        if "metadata" in data:
            metadata = data["metadata"].item()
            template.update(metadata)
        return template
    elif path.suffix == ".pt":
        return torch.load(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
