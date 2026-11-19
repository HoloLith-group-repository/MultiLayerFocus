"""Coherent isotropic planar multilayer response.

Amplitude conventions (per polarization and incidence angle, unit incident
amplitude at the entrance plane z_in):

* ``plus[j]``  forward (+z) amplitude referenced at the *bottom* boundary of
  medium j (for the incident medium: at z_in, i.e. 1; for the exit medium: at
  z_out, i.e. t).
* ``minus[j]`` backward (-z) amplitude referenced at the *top* boundary of
  medium j (for the incident medium: at z_in, i.e. r; for the exit medium: 0).

Inside a finite layer of thickness d the field is therefore
    plus[j] * exp(i kz zeta) + minus[j] * exp(i kz (d - zeta)),   0 <= zeta <= d,
which only contains decaying exponentials for absorbing or evanescent waves.
The amplitudes are obtained with an impedance-type (reflection-ratio) recursion
that never forms growing exponentials, so no cap on Im(kz d) is needed and the
internal fields stay accurate for arbitrarily thick or opaque layers.

The previous transfer-matrix routine is kept as ``multilayer_transfer_matrix``
for backward compatibility; it now returns amplitudes computed with the stable
recursion but expressed in the old convention (backward amplitudes referenced
at the bottom boundary).
"""
from dataclasses import dataclass
import numpy as np
from .model import Stack


@dataclass
class AngularResponse:
    rp: np.ndarray
    rs: np.ndarray
    tp: np.ndarray
    ts: np.ndarray
    ap_plus: np.ndarray      # forward p amplitude at the bottom of each medium
    ap_minus: np.ndarray     # backward p amplitude at the TOP of each medium
    as_plus: np.ndarray
    as_minus: np.ndarray
    theta: np.ndarray        # complex propagation angle in each medium
    kz: np.ndarray           # longitudinal wavenumber in each medium (rad/um)
    minus_reference: str = 'top'


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


def stable_multilayer(pol, stack: Stack, incident_angle, wavelength):
    """Return r, t, forward (bottom-referenced), backward (top-referenced), angles, kz."""
    n = np.asarray(stack.indices, dtype=complex)
    N = len(n)
    angles = mtm_compute_snell_angles(n, incident_angle)
    kz = 2*np.pi*n*np.cos(angles)/wavelength
    plus, minus = np.zeros(N, complex), np.zeros(N, complex)
    if N == 1:
        plus[0] = 1.
        return 0j, 1+0j, plus, minus, angles, kz
    d = np.asarray(stack.thicknesses, float)
    phase = np.ones(N, complex)                 # exp(i kz d) for finite layers, 1 otherwise
    phase[1:-1] = np.exp(1j*kz[1:-1]*d[1:-1])
    r = np.empty(N-1, complex); t = np.empty(N-1, complex)
    for j in range(N-1):
        r[j], t[j] = mtm_fresnel_coefficients(pol, n[j], n[j+1], angles[j], angles[j+1])
    # gamma_top[j]: backward/forward ratio at the top of medium j; gamma_bot[j] at its bottom
    gamma_top = np.zeros(N, complex); gamma_bot = np.zeros(N, complex)
    for j in range(N-2, -1, -1):
        g = gamma_bot[j+1]
        gamma_top[j] = (r[j] + g)/(1 + r[j]*g)
        gamma_bot[j] = gamma_top[j]*phase[j]**2 if 0 < j < N-1 else gamma_top[j]
    # forward amplitudes from the entrance plane upward
    plus[0] = 1.
    forward_top = 1.+0j
    for j in range(N-1):
        plus[j+1] = t[j]*forward_top/(1 + r[j]*gamma_bot[j+1])
        forward_top = plus[j+1]*phase[j+1] if j+1 < N-1 else plus[j+1]
        if j+1 < N-1:
            minus[j+1] = gamma_top[j+1]*forward_top
    minus[0] = gamma_top[0]
    return gamma_top[0], plus[-1], plus, minus, angles, kz


def multilayer_transfer_matrix(pol, stack: Stack, incident_angle, wavelength):
    """Backward-compatible signature: backward amplitudes referenced at the bottom.

    Computed with the stable recursion. Note that for very thick absorbing or
    evanescent layers (Im(kz d) > ~700) the bottom-referenced backward amplitude
    underflows; use ``stable_multilayer`` (top-referenced) in that case.
    """
    rg, tg, plus, minus, angles, kz = stable_multilayer(pol, stack, incident_angle, wavelength)
    old = minus.copy()
    d = np.asarray(stack.thicknesses, float)
    if len(old) > 2:
        old[1:-1] = minus[1:-1]*np.exp(1j*kz[1:-1]*d[1:-1])
    return rg, tg, plus, old, angles, kz


def gff_multilayer_response(stack: Stack, angles, wavelength):
    """Return s/p amplitudes, refraction angles and wavevectors across the stack."""
    angles = np.asarray(angles)
    shape = len(stack.indices), len(angles)
    arrays = [np.empty(len(angles), complex) for _ in range(4)]
    arrays += [np.empty(shape, complex) for _ in range(6)]
    rp, rs, tp, ts, ap, am, sp, sm, theta, kz = arrays
    for j, angle in enumerate(angles):
        rp[j], tp[j], ap[:, j], am[:, j], theta[:, j], kz[:, j] = stable_multilayer('p', stack, angle, wavelength)
        rs[j], ts[j], sp[:, j], sm[:, j], _, _ = stable_multilayer('s', stack, angle, wavelength)
    return AngularResponse(*arrays)


def power_coefficients(stack: Stack, angles, wavelength):
    """Plane-wave reflectance, transmittance and absorptance (s, p) versus incidence angle.

    Requires real incident and exit indices. Absorptance is 1 - R - T.
    """
    a = gff_multilayer_response(stack, angles, wavelength)
    n = np.asarray(stack.indices, complex)
    c1, cN = np.cos(a.theta[0]), np.cos(a.theta[-1])
    Rs, Rp = np.abs(a.rs)**2, np.abs(a.rp)**2
    Ts = np.abs(a.ts)**2*np.real(n[-1]*cN)/np.real(n[0]*c1)
    Tp = np.abs(a.tp)**2*np.real(n[-1]*np.conj(cN))/np.real(n[0]*np.conj(c1))
    return dict(Rs=Rs, Rp=Rp, Ts=Ts, Tp=Tp, As=1-Rs-Ts, Ap=1-Rp-Tp, response=a)
