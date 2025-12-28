"""MC-3GS: Molecule-Constrained Gaussian Splatting for 3D scene reconstruction."""

__version__ = "0.1.0"

from mc3gs.model.molecule_instance import MoleculeInstance
from mc3gs.model.scene import Scene
from mc3gs.model.templates import MoleculeTemplate

__all__ = [
    "MoleculeInstance",
    "MoleculeTemplate",
    "Scene",
    "__version__",
]
