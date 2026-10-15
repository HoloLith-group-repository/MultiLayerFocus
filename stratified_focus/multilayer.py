"""Coherent isotropic planar multilayer transfer matrix."""
from dataclasses import dataclass
import numpy as np
from .model import Stack


@dataclass
class AngularResponse:
    rp: np.ndarray
    rs: np.ndarray
    tp: np.ndarray
    ts: np.ndarray
    ap_plus: np.ndarray
    ap_minus: np.ndarray
    as_plus: np.ndarray
    as_minus: np.ndarray
    theta: np.ndarray
    kz: np.ndarray


def mtm_compute_snell_angles(indices, incident_angle):
    """Return refraction angles with forward propagating or decaying branches."""
    n = np.asarray(indices, dtype=complex)
    angle = np.arcsin(n[0]*np.sin(incident_angle)/n)
    ncos = n*np.cos(angle)
    forward = np.where(np.abs(ncos.imag) > 100*np.finfo(float).eps, ncos.imag > 0, ncos.real > 0)
    return np.where(forward, angle, np.pi-angle)


def mtm_fresnel_coefficients(pol, ni, nf, ti, tf):
    """Return reflection and transmission field amplitudes for s or p polarization."""
    ci, cf = np.cos(ti), np.cos(tf)
    if pol == 's':
        den = ni*ci + nf*cf
        return (ni*ci-nf*cf)/den, 2*ni*ci/den
    if pol == 'p':
        den = nf*ci + ni*cf
        return (nf*ci-ni*cf)/den, 2*ni*ci/den
    raise ValueError('Polarization must be s or p')


def multilayer_transfer_matrix(pol, stack: Stack, incident_angle, wavelength):
    """Return r, t, forward/backward amplitudes and angle/kz per medium.

    Strongly decaying internal phase is capped at Im(delta)=35 as a
    numerical approximation.
    """
    n = np.asarray(stack.indices, dtype=complex)
    angles = mtm_compute_snell_angles(n, incident_angle)
    kz = 2*np.pi*n*np.cos(angles)/wavelength
    if len(n) == 1:
        plus, minus = np.ones(1, complex), np.zeros(1, complex)
        return 0j, 1+0j, plus, minus, angles, kz
    delta = np.zeros(len(n), dtype=complex)
    delta[1:-1] = kz[1:-1]*stack.thicknesses[1:-1]
    delta[1:-1] = delta[1:-1].real + 1j*np.minimum(delta[1:-1].imag, 35.)
    r, t = [], []
    for j in range(len(n)-1):
        rj, tj = mtm_fresnel_coefficients(pol, n[j], n[j+1], angles[j], angles[j+1])
        r.append(rj)
        t.append(tj)
    def interface(j):
        return np.array([[1., r[j]], [r[j], 1.]], dtype=complex)/t[j]
    matrix = interface(0)
    for j in range(1, len(n)-1):
        matrix = matrix @ np.diag([np.exp(-1j*delta[j]), np.exp(1j*delta[j])]) @ interface(j)
    rg, tg = matrix[1, 0]/matrix[0, 0], 1/matrix[0, 0]
    plus, minus = np.zeros(len(n), complex), np.zeros(len(n), complex)
    plus[0], minus[0], plus[-1] = 1., rg, tg
    for j in range(len(n)-2, 0, -1):
        state = interface(j) @ [plus[j+1], minus[j+1]]
        plus[j], minus[j] = state * [np.exp(-1j*delta[j]), np.exp(1j*delta[j])]
    return rg, tg, plus, minus, angles, kz


def gff_multilayer_response(stack: Stack, angles, wavelength):
    """Return s/p amplitudes, refraction angles and wavevectors across the stack."""
    angles = np.asarray(angles)
    shape = len(stack.indices), len(angles)
    arrays = [np.empty(len(angles), complex) for _ in range(4)]
    arrays += [np.empty(shape, complex) for _ in range(6)]
    rp, rs, tp, ts, ap, am, sp, sm, theta, kz = arrays
    for j, angle in enumerate(angles):
        rp[j], tp[j], ap[:, j], am[:, j], theta[:, j], kz[:, j] = multilayer_transfer_matrix('p', stack, angle, wavelength)
        rs[j], ts[j], sp[:, j], sm[:, j], _, _ = multilayer_transfer_matrix('s', stack, angle, wavelength)
    return AngularResponse(*arrays)
