"""Exact azimuthal integration of the Debye integrals (core v1.2 addition).

Every plane-wave family of DebyeIntegrator (incident, reflected, transmitted, forward and backward waves in each
layer) has a vector amplitude V(phi, theta) and an axial phase that depends on theta only, while the lateral
phase is u cos(phi - varphi) with u = k1 rho sin(theta). Expanding V in a Fourier series in phi,
V = sum_m a_m(theta) exp(i m phi), the azimuthal integral is analytic (Jacobi-Anger):

    int_0^2pi V exp(i u cos(phi - varphi)) dphi = 2 pi sum_m i^m J_m(u) a_m(theta) exp(i m varphi).

A generated pupil carries only a few harmonics (|m| <= 2 for linear, circular, radial or azimuthal light, plus
|l| for a vortex; Zernike phases add a few more), so a field point costs N_m x N_theta operations instead of
N_phi x N_theta, and the result is exact in phi whatever the size of the window. Only the polar
quadrature remains, and it is refined adaptively by the caller.
"""
import numpy as np
from scipy.special import jv, j0, j1
from .field import DebyeIntegrator, _check_cancel


def bessel_table(mmax, u):
    """J_0 ... J_mmax at u (any shape) -> array (mmax+1, *u.shape).

    J_0 and J_1 come from the fast dedicated routines and higher orders from the upward recurrence, which is
    stable above the turning point u > m; the few entries below it are evaluated directly.
    """
    u = np.asarray(u, float)
    out = np.empty((mmax+1,)+u.shape)
    out[0] = j0(u)
    if mmax >= 1:
        out[1] = j1(u)
    if mmax >= 2:
        with np.errstate(divide='ignore', invalid='ignore'):
            for m in range(1, mmax):
                out[m+1] = 2*m/u*out[m] - out[m-1]
        small = u < mmax + 2
        if small.any():
            us = u[small]
            for m in range(2, mmax+1):
                out[m][small] = jv(m, us)
    return out


def azimuthal_order(ex, ey, tol=1e-13):
    """Highest azimuthal order of the Cartesian pupil fields sampled on an inclusive periodic phi grid
    (first axis), plus 2 for the projection on e_p, e_s and the vector bases."""
    spec = np.abs(np.fft.fft(ex[:-1], axis=0)) + np.abs(np.fft.fft(ey[:-1], axis=0))
    level = spec.max(axis=1)
    n = len(level)
    m = np.where(np.arange(n) < n//2, np.arange(n), n - np.arange(n))
    big = level > tol*(level.max() or 1)
    return int(m[big].max() if big.any() else 0) + 2


def fft_size(order):
    """Periodic azimuthal sample count that resolves harmonics up to `order` with a wide margin."""
    n = 16
    while n < 2*order + 8:
        n *= 2
    return n


class BesselIntegrator(DebyeIntegrator):
    """Drop-in replacement for DebyeIntegrator with the azimuthal integral done analytically.

    The configuration's azimuthal sample count must be N+1 (inclusive periodic grid, as everywhere in the
    package); the first N samples are Fourier transformed. Harmonics below `tol` times the largest one are
    dropped (for generated pupils they are rounding noise).
    """
    def __init__(self, config, pupil=None, want_h=False, tol=1e-12):
        super().__init__(config, pupil, want_h)
        nphi, ntheta = self.ex.shape
        self.nfft = nphi - 1
        self.k_lat = float(np.real(self.k1))*np.sin(self.theta)          # k1 sin(theta), conserved lateral wavenumber
        self._inv_w = 1/self.wphi[:-1]                                   # undo the azimuthal weights (first one halved)
        self._tol = tol
        regions = range(len(self.config.stack.indices))
        kinds = ('E', 'H') if want_h else ('E',)
        raw = {(r, w): [self._fourier(v) for v, _ in self._families(r, w)] for r in regions for w in kinds}
        peak = max(np.abs(a).max() for fams in raw.values() for a in fams) or 1.0
        keep = np.zeros(self.nfft, bool)
        for fams in raw.values():
            for a in fams:
                keep |= np.abs(a).max(axis=(0, 2)) > tol*peak
        idx = np.flatnonzero(keep)
        self.orders = np.where(idx < self.nfft//2, idx, idx - self.nfft)  # signed m of the kept harmonics
        self._idx = idx
        self._coef = {key: [a[:, idx, :] for a in fams] for key, fams in raw.items()}  # (3, Nm, Ntheta) each
        self._phase = {(r, w): [ph for _, ph in self._families(r, w)] for r in regions for w in kinds}
        self.i_m = (1j)**self.orders
        mid = np.abs(self.orders).max() if len(self.orders) else 0
        self.aliasing_risk = bool(mid >= self.nfft//2 - 1)                 # harmonics reach the Nyquist limit

    # --------------------------------------------------------------- set-up
    def _fourier(self, vec):
        nphi, ntheta = self.ex.shape
        V = vec.reshape(3, nphi, ntheta)[:, :-1, :]*self._inv_w[None, :, None]
        return 2*np.pi*np.fft.fft(V, axis=1)/self.nfft                    # (3, nfft, Ntheta), 2 pi a_m

    def _theta_phase(self, region, which, z):
        """exp(i axial phase) of every family at the axial positions z (1-D) -> list of (Nz, Ntheta)."""
        nphi, ntheta = self.ex.shape
        z = np.atleast_1d(np.asarray(z, float))
        out = []
        for ph in self._phase[(region, which)]:
            p = ph(z[:, None]).reshape(len(z), nphi, ntheta)[:, 0, :]
            out.append(np.exp(1j*p))
        return out

    def _bessel(self, rho):
        """J_m(k1 rho sin theta) for the kept orders: (Nm, n_unique, Ntheta) and the map back to rho,
        computed once per distinct radius."""
        rho = np.asarray(rho, float).ravel()
        r_unique, inverse = np.unique(np.round(rho, 12), return_inverse=True)
        u = r_unique[:, None]*self.k_lat[None, :]
        if not len(self.orders):
            return np.zeros((0,)+u.shape), inverse.ravel()
        mmax = int(np.abs(self.orders).max())
        table = bessel_table(mmax, u)
        sign = np.where((self.orders < 0) & (self.orders % 2 == 1), -1.0, 1.0)  # J_-m = (-1)^m J_m
        return table[np.abs(self.orders)]*sign[:, None, None], inverse.ravel()

    def _coefficients(self, z, which):
        """Combined angular coefficients C (Nz, 3, Nm, Ntheta) of all families at each z (any media)."""
        z = np.atleast_1d(np.asarray(z, float))
        medium = self._medium_index(z)
        C = np.zeros((len(z), 3, len(self.orders), len(self.theta)), complex)
        for region in np.unique(medium):
            sel = np.flatnonzero(medium == region)
            phases = self._theta_phase(region, which, z[sel])
            for a, e in zip(self._coef[(region, which)], phases):
                C[sel] += a[None]*e[:, None, None, :]
        return C

    @staticmethod
    def _radial(J, C):
        """Polar sums G[z, c, m, p] = sum_t J[m, p, t] C[z, c, m, t] as one batched matrix product per order."""
        nz, nc, nm, nt = C.shape
        B = np.ascontiguousarray(C.transpose(2, 3, 0, 1).reshape(nm, nt, nz*nc))      # (m, t, z*c)
        G = np.matmul(J, B.real) + 1j*np.matmul(J, B.imag)                           # (m, p, z*c), J is real
        return G.reshape(nm, -1, nz, nc).transpose(2, 3, 0, 1)                       # (z, c, m, p)

    def _assemble(self, G, varphi):
        """Sum the azimuthal orders: G (..., 3, Nm, P) and varphi (P,) -> (..., 3, P)."""
        rot = self.i_m[:, None]*np.exp(1j*self.orders[:, None]*varphi[None, :])   # (Nm, P)
        return self.prefactor*np.sum(G*rot, axis=-2)

    # --------------------------------------------------------------- public API (same as DebyeIntegrator)
    def evaluate(self, x, y, z, chunk=4096, which='E'):
        if which == 'H' and not self.want_h:
            raise ValueError('Create the integrator with want_h=True to evaluate H')
        xx, yy, zz = np.broadcast_arrays(x, y, z)
        xf, yf, zf = xx.ravel().astype(float), yy.ravel().astype(float), zz.ravel().astype(float)
        out = np.empty((3, xf.size), complex)
        for s in range(0, xf.size, chunk):
            sl = slice(s, s+chunk)
            rho, vphi = np.hypot(xf[sl], yf[sl]), np.arctan2(yf[sl], xf[sl])
            J, inv = self._bessel(rho)
            C = self._coefficients(zf[sl], which)                       # (P, 3, Nm, Nt)
            G = np.einsum('mpt,pcmt->cmp', J[:, inv, :], C)
            out[:, sl] = self._assemble(G, vphi)
        return out.reshape((3,)+xx.shape)

    def generate_focal_field_xy(self, x, y, z, progress=None, cancel=None, which='E'):
        x, y = np.asarray(x, float), np.asarray(y, float)
        X, Y = np.meshgrid(x, y)
        J, inv = self._bessel(np.hypot(X, Y))
        _check_cancel(cancel)
        C = self._coefficients([float(z)], which)                        # (1, 3, Nm, Nt)
        G = self._radial(J, C)[0][:, :, inv]                             # (3, Nm, P)
        out = self._assemble(G, np.arctan2(Y, X).ravel())
        if progress: progress(1.0)
        return out.reshape(3, len(y), len(x))

    def generate_focal_field_axial(self, x, y, z, x0, y0, progress=None, chunk=64, cancel=None, which='E'):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        lines = ((x, np.full_like(x, y0)), (np.full_like(y, x0), y))
        out = []
        for k, (lx, ly) in enumerate(lines):
            J, inv = self._bessel(np.hypot(lx, ly)); J = J[:, inv, :]
            vphi = np.arctan2(ly, lx)
            F = np.empty((3, len(z), len(lx)), complex)
            for s in range(0, len(z), chunk):
                _check_cancel(cancel)
                C = self._coefficients(z[s:s+chunk], which)              # (nz, 3, Nm, Nt)
                G = self._radial(J, C)
                F[:, s:s+chunk] = np.moveaxis(self._assemble(G, vphi), 0, 1)
                if progress: progress((k + min(1, (s+chunk)/len(z)))/2)
            out.append(F)
        return out[0], out[1]

    def generate_focal_field_volume(self, x, y, z, progress=None, phase_cache_bytes=None, cancel=None, which='E',
                                    chunk=16):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        if 16*3*len(x)*len(y)*len(z) > 1024**3:
            raise ValueError('Volume complex fields exceed the 1 GiB allocation limit')
        X, Y = np.meshgrid(x, y)
        J, inv = self._bessel(np.hypot(X, Y))
        vphi = np.arctan2(Y, X).ravel()
        out = np.empty((3, len(z), len(y), len(x)), complex)
        for s in range(0, len(z), chunk):
            _check_cancel(cancel)
            C = self._coefficients(z[s:s+chunk], which)
            G = self._radial(J, C)[..., inv]
            out[:, s:s+chunk] = np.moveaxis(self._assemble(G, vphi), 0, 1).reshape(3, -1, len(y), len(x))
            if progress: progress(min(1.0, (s+chunk)/len(z)))
        return out
