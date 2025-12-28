"""Rendering utilities for Gaussian splatting."""

from mc3gs.render.backend import RenderBackend, get_backend
from mc3gs.render.projection import project_gaussians, projection_jacobian
from mc3gs.render.sh import eval_sh_basis, shade_sh
from mc3gs.render.splat_renderer import ReferenceSplatRenderer

__all__ = [
    "ReferenceSplatRenderer",
    "RenderBackend",
    "eval_sh_basis",
    "get_backend",
    "project_gaussians",
    "projection_jacobian",
    "shade_sh",
]
