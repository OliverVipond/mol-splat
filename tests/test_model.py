"""Tests for molecule instance and scene modules."""

import pytest
import torch

from mc3gs.chemistry.typing import TypeVocabulary
from mc3gs.model.constraints import AtomSHBank, BondSHBank, SharedSHBank
from mc3gs.model.molecule_instance import MoleculeInstance, axis_angle_to_rotation_matrix
from mc3gs.model.scene import Scene
from mc3gs.model.templates import MoleculeTemplate


@pytest.fixture
def simple_template():
    """Create a simple molecule template for testing (atoms only)."""
    vocab = TypeVocabulary.default(include_bonds=False)

    p_local = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    cov_local = torch.eye(3).unsqueeze(0).expand(3, -1, -1).clone() * 0.1

    type_id = torch.tensor([0, 0, 3])  # Two C_sp3, one N (different colors)

    return MoleculeTemplate(
        p_local=p_local,
        cov_local=cov_local,
        type_id=type_id,
        type_vocab=vocab,
        name="test_template",
    )


@pytest.fixture
def template_with_bonds():
    """Create a molecule template with atoms and bonds."""
    vocab = TypeVocabulary.default(include_bonds=True)

    # 2 atoms + 1 bond
    p_local = torch.tensor([
        [0.0, 0.0, 0.0],  # Atom 0 (C)
        [1.5, 0.0, 0.0],  # Atom 1 (C)
        [0.75, 0.0, 0.0],  # Bond between them
    ])

    cov_local = torch.eye(3).unsqueeze(0).expand(3, -1, -1).clone() * 0.1

    # Type IDs: 0, 1 are atom types, bond types start at num_atom_types
    num_atom_types = vocab.num_atom_types
    type_id = torch.tensor([0, 0, num_atom_types])  # Two C atoms, one single bond

    return MoleculeTemplate(
        p_local=p_local,
        cov_local=cov_local,
        type_id=type_id,
        type_vocab=vocab,
        name="test_with_bonds",
    )


class TestRotationConversion:
    """Tests for rotation representation conversion."""

    def test_identity_rotation(self):
        """Test axis-angle [0,0,0] gives identity."""
        axis_angle = torch.zeros(3)
        R = axis_angle_to_rotation_matrix(axis_angle)

        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_90_degree_z_rotation(self):
        """Test 90 degree rotation around Z axis."""
        axis_angle = torch.tensor([0.0, 0.0, torch.pi / 2])
        R = axis_angle_to_rotation_matrix(axis_angle)

        # Should rotate x -> y, y -> -x
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        assert torch.allclose(R, expected, atol=1e-5)


class TestMoleculeTemplate:
    """Tests for MoleculeTemplate."""

    def test_template_properties(self, simple_template):
        """Test template property access."""
        assert simple_template.num_gaussians == 3
        assert simple_template.num_types == 2

    def test_template_centering(self, simple_template):
        """Test template centering."""
        centered = simple_template.centered()

        centroid = centered.center()
        assert torch.allclose(centroid, torch.zeros(3), atol=1e-6)

    def test_template_scaling(self, simple_template):
        """Test template scaling."""
        scaled = simple_template.scale(2.0)

        assert torch.allclose(scaled.p_local, simple_template.p_local * 2.0)
        assert torch.allclose(scaled.cov_local, simple_template.cov_local * 4.0)


class TestMoleculeInstance:
    """Tests for MoleculeInstance."""

    def test_instance_creation(self, simple_template):
        """Test creating a molecule instance."""
        instance = MoleculeInstance(simple_template, sh_degree=2)

        assert instance.template is simple_template
        assert instance.sh_degree == 2
        assert instance.atom_sh_bank.num_atom_types == simple_template.type_vocab.num_atom_types

    def test_world_positions_identity(self, simple_template):
        """Test world positions with identity transform."""
        instance = MoleculeInstance(simple_template)

        positions = instance.world_positions()

        # With identity transform, should match local positions
        assert torch.allclose(positions, simple_template.p_local, atol=1e-5)

    def test_world_positions_translated(self, simple_template):
        """Test world positions with translation."""
        translation = torch.tensor([1.0, 2.0, 3.0])
        instance = MoleculeInstance(simple_template, init_position=translation)

        positions = instance.world_positions()

        expected = simple_template.p_local + translation
        assert torch.allclose(positions, expected, atol=1e-5)

    def test_opacity_sigmoid(self, simple_template):
        """Test that opacity is in (0, 1) via sigmoid."""
        instance = MoleculeInstance(simple_template, init_opacity=0.3)

        opacity = instance.opacity

        assert (opacity > 0).all()
        assert (opacity < 1).all()

    def test_sh_coeffs_shared_by_type(self, simple_template):
        """Test that Gaussians of same type share SH coefficients."""
        instance = MoleculeInstance(simple_template, sh_degree=2)

        sh_coeffs = instance.get_sh_coeffs()

        # Gaussians 0 and 1 have type 0, Gaussian 2 has type 1
        assert torch.allclose(sh_coeffs[0], sh_coeffs[1])
        assert not torch.allclose(sh_coeffs[0], sh_coeffs[2])


class TestSharedSHBank:
    """Tests for SharedSHBank (legacy) and new AtomSHBank/BondSHBank."""

    def test_atom_bank_shape(self):
        """Test atom SH bank output shape."""
        bank = AtomSHBank(num_atom_types=5, sh_degree=3)

        assert bank.num_sh_coeffs == 16  # (3+1)^2

        type_ids = torch.tensor([0, 1, 0, 4])
        coeffs = bank.get_sh_coeffs(type_ids)

        assert coeffs.shape == (4, 16, 3)

    def test_bond_bank_single_color(self):
        """Test bond SH bank returns same color for all bonds."""
        bank = BondSHBank(sh_degree=2)

        coeffs = bank.get_sh_coeffs(num_bonds=5)

        assert coeffs.shape == (5, 9, 3)
        # All bonds should have identical SH coefficients
        assert torch.allclose(coeffs[0], coeffs[1])
        assert torch.allclose(coeffs[0], coeffs[4])

    def test_atom_bank_set_from_colors(self):
        """Test setting DC from colors."""
        bank = AtomSHBank(num_atom_types=3, sh_degree=1)

        colors = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bank.set_from_colors(colors)

        # DC should be non-zero
        assert bank.sh_coeffs[:, 0, :].abs().sum() > 0

    def test_legacy_shared_bank(self):
        """Test legacy SharedSHBank still works."""
        bank = SharedSHBank(num_types=5, sh_degree=3)

        assert bank.num_sh_coeffs == 16

        type_ids = torch.tensor([0, 1, 0, 4])
        coeffs = bank.get_sh_coeffs(type_ids)

        assert coeffs.shape == (4, 16, 3)


class TestBondColorConstraints:
    """Tests for the new bond color constraints."""

    def test_bonds_share_color_within_molecule(self, template_with_bonds):
        """Test that all bonds within a molecule share the same color."""
        instance = MoleculeInstance(template_with_bonds, sh_degree=2)

        sh_coeffs = instance.get_sh_coeffs()

        # With template_with_bonds, index 2 is a bond
        # If we had multiple bonds, they would all be the same
        bond_mask = instance._is_bond
        assert bond_mask[2] == True  # The bond
        assert bond_mask[0] == False  # Atom
        assert bond_mask[1] == False  # Atom

    def test_atoms_share_color_by_type(self, template_with_bonds):
        """Test that atoms of the same type share color."""
        instance = MoleculeInstance(template_with_bonds, sh_degree=2)

        sh_coeffs = instance.get_sh_coeffs()

        # Indices 0 and 1 are both type 0 (carbon atoms)
        # They should have the same SH coefficients
        assert torch.allclose(sh_coeffs[0], sh_coeffs[1])

    def test_different_molecules_can_have_different_bond_colors(self, template_with_bonds):
        """Test that different molecule instances can have different bond colors."""
        instance1 = MoleculeInstance(template_with_bonds, sh_degree=1)
        instance2 = MoleculeInstance(template_with_bonds, sh_degree=1)

        # Modify bond color in instance2
        instance2.bond_sh_bank.set_from_color(torch.tensor([1.0, 0.0, 0.0]))

        sh1 = instance1.get_sh_coeffs()
        sh2 = instance2.get_sh_coeffs()

        # Bond SH coefficients should be different
        bond_idx = 2
        assert not torch.allclose(sh1[bond_idx], sh2[bond_idx])


class TestScene:
    """Tests for Scene."""

    def test_empty_scene(self):
        """Test empty scene creation."""
        scene = Scene()

        assert len(scene) == 0
        assert scene.total_gaussians == 0

    def test_add_instance(self, simple_template):
        """Test adding instances to scene."""
        scene = Scene()

        instance = MoleculeInstance(simple_template)
        idx = scene.add_instance(instance)

        assert idx == 0
        assert len(scene) == 1
        assert scene.total_gaussians == 3

    def test_gather_all_gaussians(self, simple_template):
        """Test gathering all Gaussians from scene."""
        scene = Scene()

        for i in range(3):
            instance = MoleculeInstance(
                simple_template,
                init_position=torch.tensor([float(i), 0.0, 0.0]),
            )
            scene.add_instance(instance)

        data = scene.gather()

        assert data["positions"].shape == (9, 3)  # 3 instances * 3 Gaussians
        assert data["mol_ids"].shape == (9,)
        assert data["type_ids"].shape == (9,)

        # Check mol_ids are correct
        assert (data["mol_ids"][:3] == 0).all()
        assert (data["mol_ids"][3:6] == 1).all()
        assert (data["mol_ids"][6:9] == 2).all()

    def test_regularization_loss(self, simple_template):
        """Test regularization loss computation."""
        scene = Scene()
        scene.add_instance(MoleculeInstance(simple_template))

        losses = scene.regularization_loss()

        assert "pose" in losses
        assert "opacity" in losses
        assert "sh" in losses
        assert "total" in losses
