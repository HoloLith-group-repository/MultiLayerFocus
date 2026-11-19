"""Field normalization and local three-dimensional polarization ellipses."""
from dataclasses import dataclass
import numpy as np


def intensity(field):
    return np.sum(np.abs(field)**2, axis=0)


def normalize_focal_fields(xy, lateral_range, *other):
    """Scale all supplied fields by the integrated XY intensity."""
    ny, nx = xy.shape[1:]
    wx = np.ones(nx); wx[[0, -1]] = 0.5
    wy = np.ones(ny); wy[[0, -1]] = 0.5
    integral = np.sum(intensity(xy)*wy[:, None]*wx[None, :]) * lateral_range**2/((nx-1)*(ny-1))
    if integral <= 0 or not np.isfinite(integral):
        raise ValueError('XY field has zero or nonfinite integrated intensity')
    factor = np.sqrt(integral)
    return (xy/factor, *(field/factor for field in other)), factor


@dataclass
class Polarization:
    intensity: np.ndarray
    psi: np.ndarray
    rectifying_phase: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    normal: np.ndarray
    ellipticity: np.ndarray
    valid: np.ndarray
    circular: np.ndarray


def polarization_ellipse_3d(field, threshold=0.05):
    """Return polarization ellipse axes and masks for a (3, ...) complex field."""
    if not 0 <= threshold <= 1:
        raise ValueError('Relative intensity threshold must lie in [0, 1]')
    inten = intensity(field)
    psi = np.sum(field**2, axis=0)
    valid = inten >= threshold*inten.max() if inten.max() > 0 else np.zeros_like(inten, bool)
    circular = valid & (np.abs(psi) <= 1e-8*inten)
    gamma = np.ones_like(psi)
    active = valid & ~circular
    gamma[active] = np.exp(-0.5j*np.angle(psi[active]))
    rotated = field*gamma[None, ...]
    major = np.where(valid[None, ...], rotated.real, 0.)
    minor = np.where(valid[None, ...], rotated.imag, 0.)
    normal = np.moveaxis(np.cross(np.moveaxis(major, 0, -1), np.moveaxis(minor, 0, -1)), -1, 0)
    a, b = np.linalg.norm(major, axis=0), np.linalg.norm(minor, axis=0)
    ratio = np.divide(b, a, out=np.zeros_like(a), where=valid & (a > 0))
    phase = np.where(valid & ~circular, np.angle(gamma), np.nan)
    return Polarization(inten, psi, phase, major, minor, normal, ratio, valid, circular)
