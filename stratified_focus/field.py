"""Vector Debye integration across planar stacks.

Version 1.1 changes (backward compatible API):
* internal backward waves are referenced at the top of each layer (stable for
  opaque and evanescent layers, see multilayer.py);
* optional Gauss-Legendre polar quadrature (Sampling.quadrature = 'gauss');
* optional magnetic field (in units of Z0*H, i.e. same units as E) for Poynting
  vector, energy-flux and absorption analysis;
* sampling-error estimator and absolute (power-calibrated) field scaling.
"""
from dataclasses import dataclass, replace
from time import perf_counter
from collections.abc import Callable
import numpy as np
from .model import SimulationConfig
from .pupil import generate_inc_field, generate_pupil_coordinates, polar_nodes, azimuthal_nodes, critical_angles
from .multilayer import gff_multilayer_response, AngularResponse
from .analysis import normalize_focal_fields, polarization_ellipse_3d, Polarization

C0 = 299792458.0
EPS0 = 8.8541878128e-12
Z0 = 376.730313668


class SimulationCancelled(Exception):
    """A calculation stopped at a cooperative cancellation checkpoint."""


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise SimulationCancelled('Calculation aborted')


def _stage_callback(progress, stage_progress, name, total, offset, weight):
    if stage_progress:
        stage_progress(name,0,total)
    def update(fraction):
        if progress:
            progress(offset+weight*fraction)
        if stage_progress:
            stage_progress(name,min(total,round(fraction*total)),total)
    return update


@dataclass
class Result:
    config: SimulationConfig
    pupil_ex: np.ndarray
    pupil_ey: np.ndarray
    pupil_x: np.ndarray
    pupil_y: np.ndarray
    angular: AngularResponse
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    xy: np.ndarray                  # (component, y, x)
    xz: np.ndarray | None          # (component, z, x), z descending
    yz: np.ndarray | None          # (component, z, y), z descending
    normalization: float
    polarization: Polarization
    elapsed_seconds: float | None = None
    volume: np.ndarray | None = None       # raw complex (component, z, y, x), ascending z
    volume_x: np.ndarray | None = None
    volume_y: np.ndarray | None = None
    volume_z: np.ndarray | None = None
    volume_seconds: float | None = None
    sampling_error: float | None = None    # estimated relative angular-sampling error


class DebyeIntegrator:
    """Precompute pupil and stack terms, then integrate arbitrary point sets.

    Arrays have phi as their first angular axis and theta as the second.
    At the first interface the incident side is selected; at the last,
    the exit side is selected; at internal interfaces the upper medium.
    """
    def __init__(self, config, pupil=None, want_h=False):
        config.validate()
        self.config = config
        obj, light, stack, grid = config.objective, config.illumination, config.stack, config.sampling
        quad = getattr(grid, 'quadrature', 'trapezoid')
        self.breakpoints = critical_angles(stack, float(np.real(stack.indices[0])), obj.theta_max) if quad == 'gauss' else ()
        self.ex, self.ey, self.xp, self.yp = pupil if pupil is not None else generate_inc_field(
            obj, light, grid.phi, grid.theta, quad, self.breakpoints)
        if self.ex.shape != (grid.phi, grid.theta) or self.ey.shape != self.ex.shape:
            raise ValueError('Pupil fields must have shape (phi samples, theta samples)')
        self.ex, self.ey = np.asarray(self.ex, complex), np.asarray(self.ey, complex)
        self.k1 = 2*np.pi*stack.indices[0]/light.wavelength
        self.f = obj.focal_length
        self.want_h = want_h
        theta, wtheta = polar_nodes(obj.theta_max, grid.theta, quad, self.breakpoints)
        phi, wphi = azimuthal_nodes(grid.phi)
        self.theta, self.phi, self.wtheta, self.wphi = theta, phi, wtheta, wphi
        self.angular = gff_multilayer_response(stack, theta, light.wavelength)
        th, ph = np.meshgrid(theta, phi)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        weight = np.sqrt(ct)*st*wphi[:, None]*wtheta[None, :]
        ep = self.ex*cp + self.ey*sp
        es = -self.ex*sp + self.ey*cp
        self.ep, self.es, self.weight = ep, es, weight
        self.sp, self.cp = sp, cp
        self.st, self.ct = st, ct
        self.lx = (self.k1*st*cp).ravel()
        self.ly = (self.k1*st*sp).ravel()
        self.kz_i = (self.k1*ct).ravel()
        self.prefactor = -1j*self.k1*self.f*np.exp(-1j*self.k1*self.f)/(2*np.pi)
        ar = self.angular
        n = np.asarray(stack.indices, complex)
        N = len(n)
        one = np.ones_like(st)
        # vectors in the two outer media (incident + reflected, transmitted)
        self.v_i, self.h_i = self._vectors(one*1, one*0, one*1, one*0, ct, st, n[0], forward=True)
        self.v_r, self.h_r = self._vectors(one*0, ar.rp[None, :]*one, one*0, ar.rs[None, :]*one,
                                           ct, st, n[0], forward=False)
        ctrans, strans = np.cos(ar.theta[-1])[None, :]*one, np.sin(ar.theta[-1])[None, :]*one
        self.v_t, self.h_t = self._vectors(ar.tp[None, :]*one, one*0, ar.ts[None, :]*one, one*0,
                                           ctrans, strans, n[-1], forward=True)
        self.kz_t = np.broadcast_to(ar.kz[-1], st.shape).ravel()
        self._internal_cache = {}

    # --------------------------------------------------------------- vectors
    def _vectors(self, pp, pm, sp_, sm_, ct, st, nj, forward=True):
        """E and Z0*H vector coefficients (3, M) of one plane-wave family.

        pp/sp_ are forward p/s amplitudes, pm/sm_ backward ones; only one direction is
        used per family (forward=True -> pp, sp_; False -> pm, sm_).
        """
        cp, sp = self.cp, self.sp
        if forward:
            P, S, sgn = self.ep*pp, self.es*sp_, 1
        else:
            P, S, sgn = self.ep*pm, self.es*sm_, -1
        # e_s = (-sin p, cos p, 0); e_p(+/-) = (+/-cos t cos p, +/-cos t sin p, -sin t)
        epx, epy, epz = sgn*ct*cp, sgn*ct*sp, -st
        esx, esy = -sp, cp
        E = np.stack((S*esx + P*epx, S*esy + P*epy, P*epz)).reshape(3, -1)*self.weight.ravel()
        if not self.want_h:
            return E, None
        # Z0 H = n (P e_s - S e_p) for both directions
        H = nj*np.stack((P*esx - S*epx, P*esy - S*epy, -S*epz)).reshape(3, -1)*self.weight.ravel()
        return E, H

    def _internal_vectors(self, medium):
        if medium not in self._internal_cache:
            ar = self.angular
            one = np.ones_like(self.st)
            ct, st = np.cos(ar.theta[medium])[None, :]*one, np.sin(ar.theta[medium])[None, :]*one
            nj = complex(self.config.stack.indices[medium])
            plus = self._vectors(ar.ap_plus[medium][None, :]*one, None, ar.as_plus[medium][None, :]*one, None,
                                 ct, st, nj, forward=True)
            minus = self._vectors(None, ar.ap_minus[medium][None, :]*one, None, ar.as_minus[medium][None, :]*one,
                                  ct, st, nj, forward=False)
            kz = np.broadcast_to(ar.kz[medium], self.st.shape).ravel()
            self._internal_cache[medium] = plus, minus, kz
        return self._internal_cache[medium]

    # --------------------------------------------------------------- families
    def _families(self, region, which):
        """List of (vector (3,M), axial-phase function of z (M,) -> phase) for one medium."""
        stack = self.config.stack
        pick = (lambda pair: pair[0]) if which == 'E' else (lambda pair: pair[1])
        N = len(stack.indices)
        if region == 0:
            fam = [((self.v_i, self.h_i), lambda z: z*self.kz_i)]
            if N > 1:
                fam.append(((self.v_r, self.h_r), lambda z: (2*stack.z_in - z)*self.kz_i))
        elif region == N-1:
            fam = [((self.v_t, self.h_t), lambda z: stack.z_in*self.kz_i + (z - stack.z_out)*self.kz_t)]
        else:
            plus, minus, kz = self._internal_vectors(region)
            zb = stack.interfaces[region-1]; d = stack.thicknesses[region]
            base = stack.z_in*self.kz_i
            fam = [(plus, lambda z: base + (z - zb)*kz),
                   (minus, lambda z: base + (d - (z - zb))*kz)]
        return [(pick(v), phase) for v, phase in fam]

    def _medium_index(self, zf):
        stack = self.config.stack
        if np.all(zf == zf[0]):
            return np.full(len(zf), stack.locate_medium(float(zf[0]))[0], dtype=int)
        medium = np.searchsorted(stack.interfaces, zf, side='right')
        medium[zf <= stack.z_in] = 0
        medium[zf >= stack.z_out] = len(stack.indices)-1
        return medium

    def evaluate(self, x, y, z, chunk=64, which='E'):
        """Pointwise complex (Ex,Ey,Ez) (or Z0*H with which='H'), broadcastable x/y/z."""
        if which == 'H' and not self.want_h:
            raise ValueError('Create the integrator with want_h=True to evaluate H')
        xx, yy, zz = np.broadcast_arrays(x, y, z)
        output = np.empty((3, xx.size), complex)
        xf, yf, zf = xx.ravel(), yy.ravel(), zz.ravel()
        medium = self._medium_index(zf)
        for region in np.unique(medium):
            ids = np.flatnonzero(medium == region)
            fams = self._families(region, which)
            for start in range(0, len(ids), chunk):
                sel = ids[start:start+chunk]
                lateral = xf[sel, None]*self.lx + yf[sel, None]*self.ly
                zl = zf[sel, None]
                value = 0
                for v, phase in fams:
                    value = value + v@np.exp(1j*(lateral + phase(zl))).T
                output[:, sel] = self.prefactor*value
        return output.reshape((3,)+xx.shape)

    def _angular_coefficients(self, z, which='E'):
        """Weighted angular vector coefficients at a single axial coordinate."""
        region, _ = self.config.stack.locate_medium(float(z))
        total = 0
        for v, phase in self._families(region, which):
            total = total + v*np.exp(1j*phase(float(z)))[None, :]
        return total

    def generate_focal_field_xy(self, x, y, z, progress=None, cancel=None, which='E'):
        """Integrate a regular XY plane using reusable lateral phase matrices."""
        x, y = np.asarray(x), np.asarray(y)
        np_phi, np_theta = self.ex.shape
        coefficients = self._angular_coefficients(z, which).reshape(3,np_phi,np_theta)
        lx = self.lx.reshape(np_phi,np_theta)
        ly = self.ly.reshape(np_phi,np_theta)
        field = np.zeros((3,len(y),len(x)),complex)
        for t in range(np_theta):
            _check_cancel(cancel)
            x_phase = np.exp(1j*np.outer(lx[:,t],x))
            y_phase = np.exp(1j*np.outer(y,ly[:,t]))
            for component in range(3):
                field[component] += y_phase @ (coefficients[component,:,t,None]*x_phase)
            if progress:
                progress((t+1)/np_theta)
        _check_cancel(cancel)
        return self.prefactor*field

    def generate_focal_field_volume(self, x, y, z, progress=None, phase_cache_bytes=128*1024**2, cancel=None, which='E'):
        """Evaluate a raw complex XYZ volume; reuse bounded lateral phase caches."""
        x, y, z = map(np.asarray, (x, y, z))
        shape = (3, len(z), len(y), len(x))
        if 16*np.prod(shape, dtype=np.int64) > 1024**3:
            raise ValueError('Volume complex fields exceed the 1 GiB allocation limit')
        nphi, ntheta = self.ex.shape
        lx = self.lx.reshape(nphi, ntheta)
        ly = self.ly.reshape(nphi, ntheta)
        cache_size = 16*nphi*ntheta*(len(x)+len(y))
        phases = None
        if cache_size <= phase_cache_bytes:
            phases = []
            for t in range(ntheta):
                _check_cancel(cancel)
                phases.append((np.exp(1j*np.outer(lx[:,t],x)),np.exp(1j*np.outer(y,ly[:,t]))))
        fields = np.empty(shape, complex)
        for q, zi in enumerate(z):
            _check_cancel(cancel)
            coefficients = self._angular_coefficients(float(zi), which).reshape(3, nphi, ntheta)
            plane = fields[:, q]; plane.fill(0)
            for t in range(ntheta):
                xp, yp = phases[t] if phases is not None else (
                    np.exp(1j*np.outer(lx[:,t], x)), np.exp(1j*np.outer(y, ly[:,t])))
                for c in range(3):
                    plane[c] += yp @ (coefficients[c,:,t,None]*xp)
            plane *= self.prefactor
            if progress:
                progress((q+1)/len(z))
        _check_cancel(cancel)
        return fields

    def generate_focal_field_axial(self, x, y, z, x0, y0, progress=None, chunk=8, cancel=None, which='E'):
        """Integrate XZ and YZ planes while reusing lateral phase matrices."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        x_phase = np.exp(1j*(self.lx[:,None]*x[None,:] + self.ly[:,None]*y0))
        y_phase = np.exp(1j*(self.lx[:,None]*x0 + self.ly[:,None]*y[None,:]))
        xz = np.empty((3,len(z),len(x)),complex)
        yz = np.empty((3,len(z),len(y)),complex)
        for start in range(0,len(z),chunk):
            _check_cancel(cancel)
            stop = min(start+chunk,len(z))
            coefficients = np.stack([self._angular_coefficients(zi, which) for zi in z[start:stop]])
            xz[:,start:stop] = np.moveaxis(coefficients @ x_phase,0,1)*self.prefactor
            _check_cancel(cancel)
            yz[:,start:stop] = np.moveaxis(coefficients @ y_phase,0,1)*self.prefactor
            if progress:
                progress(stop/len(z))
        _check_cancel(cancel)
        return xz, yz

    # --------------------------------------------------------------- physics helpers
    def pupil_power_integral(self):
        """int |E_pupil|^2 dA over the aperture (um^2 x pupil-amplitude^2) with the solver's quadrature."""
        f = self.f
        w = self.wphi[:, None]*self.wtheta[None, :]*f**2*np.sin(self.theta)[None, :]*np.cos(self.theta)[None, :]
        return float(np.sum((np.abs(self.ex)**2 + np.abs(self.ey)**2)*w))

    def absolute_scale(self, power_watt, pupil_index=1.0):
        """Factor converting raw solver fields into V/m for a given power in the aperture.

        Raw fields correspond to the pupil amplitude unit; the energy-conserving aplanatic
        factor sqrt(n_pupil/n1) omitted by the solver is applied here.
        """
        n1 = float(np.real(self.config.stack.indices[0]))
        area = self.pupil_power_integral()*1e-12              # m^2 * amplitude^2
        e0 = np.sqrt(2*power_watt/(C0*EPS0*pupil_index*area))  # V/m per pupil amplitude unit
        return e0*np.sqrt(pupil_index/n1)


def poynting(E, H):
    """Time-averaged Poynting vector 0.5*Re(E x H*) from E (V/m) and Z0*H (V/m) arrays (3, ...), in W/m^2."""
    Hc = np.conj(H)
    return 0.5*np.real(np.stack((E[1]*Hc[2]-E[2]*Hc[1], E[2]*Hc[0]-E[0]*Hc[2], E[0]*Hc[1]-E[1]*Hc[0])))/Z0


def absorbed_power_density(E, n, wavelength_um):
    """Absorbed power density 0.5*omega*eps0*Im(eps)*|E|^2 in W/m^3 for E in V/m."""
    omega = 2*np.pi*C0/(wavelength_um*1e-6)
    eps_im = np.imag(complex(n)**2)
    return 0.5*omega*EPS0*eps_im*np.sum(np.abs(E)**2, axis=0)


def estimate_sampling_error(config, probe=None, factor=2, npts=6):
    """Estimate the relative angular-sampling error by comparing with a refined grid.

    Probe points cover the XY window at z0 and, when requested, the axial range.
    Only valid for internally generated pupils. Returns max|E_N - E_2N| / max|E_2N|.
    """
    g = config.sampling
    if probe is None:
        xs = np.linspace(g.x0-g.lateral_range/2, g.x0+g.lateral_range/2, npts)
        ys = np.linspace(g.y0-g.lateral_range/2, g.y0+g.lateral_range/2, npts)
        X, Y = np.meshgrid(xs, ys)
        pts = [np.c_[X.ravel(), Y.ravel(), np.full(X.size, g.z0)]]
        if g.compute_axial or g.compute_volume:
            zs = np.linspace(g.z0-g.axial_range/2, g.z0+g.axial_range/2, 3*npts)
            pts.append(np.c_[np.full(zs.size, g.x0), np.full(zs.size, g.y0), zs])
            pts.append(np.c_[np.full(zs.size, g.x0+g.lateral_range/4), np.full(zs.size, g.y0), zs])
        probe = np.vstack(pts)
    fine = replace(config, sampling=replace(g, phi=int(factor*(g.phi-1))+1, theta=int(factor*g.theta)))
    a = DebyeIntegrator(config).evaluate(probe[:, 0], probe[:, 1], probe[:, 2])
    b = DebyeIntegrator(fine).evaluate(probe[:, 0], probe[:, 1], probe[:, 2])
    scale = np.sqrt(np.max(np.sum(np.abs(b)**2, axis=0)))
    if scale == 0 or not np.isfinite(scale):
        return float('nan')
    return float(np.sqrt(np.max(np.sum(np.abs(a-b)**2, axis=0)))/scale)


def simulate(config: SimulationConfig, pupil=None, progress: Callable[[float], None] | None=None,
             stage_progress=None, cancel=None, check_sampling=False):
    """Compute selected fields; report overall fractions and named stage counts."""
    started = perf_counter(); _check_cancel(cancel)
    if stage_progress: stage_progress('Preparing pupil and multilayer response',0,1)
    integrator = DebyeIntegrator(config,pupil)
    _check_cancel(cancel)
    if stage_progress: stage_progress('Preparing pupil and multilayer response',1,1)
    g = config.sampling
    units = 1+int(g.compute_axial)+(g.volume_z if g.compute_volume else 0)
    field_progress = (lambda value:progress(.99*value)) if progress else None
    x = np.linspace(g.x0-g.lateral_range/2,g.x0+g.lateral_range/2,g.xy)
    y = np.linspace(g.y0-g.lateral_range/2,g.y0+g.lateral_range/2,g.xy)
    z = np.linspace(g.z0+g.axial_range/2,g.z0-g.axial_range/2,g.axial)
    xy = integrator.generate_focal_field_xy(x,y,g.z0,
        _stage_callback(field_progress,stage_progress,'XY transverse field',g.theta,0,1/units),cancel=cancel)
    xz = yz = None
    if g.compute_axial:
        xz,yz = integrator.generate_focal_field_axial(x,y,z,g.x0,g.y0,
            _stage_callback(field_progress,stage_progress,'XZ and YZ sections',g.axial,1/units,1/units),cancel=cancel)
    volume = vx = vy = vz = volume_seconds = None
    if g.compute_volume:
        volume_started = perf_counter()
        vx = np.linspace(g.x0-g.lateral_range/2,g.x0+g.lateral_range/2,g.volume_xy)
        vy = np.linspace(g.y0-g.lateral_range/2,g.y0+g.lateral_range/2,g.volume_xy)
        vz = np.linspace(g.z0-g.axial_range/2,g.z0+g.axial_range/2,g.volume_z)
        volume = integrator.generate_focal_field_volume(vx,vy,vz,
            _stage_callback(field_progress,stage_progress,'XYZ volume',g.volume_z,
                            (1+int(g.compute_axial))/units,g.volume_z/units),cancel=cancel)
        volume_seconds = perf_counter()-volume_started
    _check_cancel(cancel)
    if stage_progress: stage_progress('Normalization and polarization',0,1)
    fields,factor = normalize_focal_fields(xy,g.lateral_range,*(v for v in (xz,yz) if v is not None))
    xy = fields[0]
    if g.compute_axial: xz,yz = fields[1:]
    polarization = polarization_ellipse_3d(xy)
    sampling_error = None
    if check_sampling and pupil is None:
        if stage_progress: stage_progress('Estimating angular sampling error',0,1)
        sampling_error = estimate_sampling_error(config)
    _check_cancel(cancel)
    if stage_progress: stage_progress('Normalization and polarization',1,1)
    if progress: progress(1.)
    return Result(config,integrator.ex,integrator.ey,integrator.xp,integrator.yp,
                  integrator.angular,x,y,z,xy,xz,yz,factor,polarization,
                  perf_counter()-started,volume,vx,vy,vz,volume_seconds,sampling_error)


def compute_additional(result, *, axial=False, volume=False, progress=None, stage_progress=None, cancel=None):
    """Add missing outputs atomically using the stored configuration and pupil.

    Existing XY fields, polarization and normalization are retained. Axial fields
    use that normalization; volume fields retain their raw complex amplitudes.
    """
    started = perf_counter(); _check_cancel(cancel)
    axial = axial and (result.xz is None or result.yz is None)
    volume = volume and result.volume is None
    if not axial and not volume:
        return result
    g = result.config.sampling
    sampling = replace(g,compute_axial=g.compute_axial or axial,compute_volume=g.compute_volume or volume)
    config = replace(result.config,sampling=sampling)
    config.validate()
    if not np.isfinite(result.normalization) or result.normalization <= 0:
        raise ValueError('The stored XY normalization is invalid; run a new simulation')
    if stage_progress: stage_progress('Preparing stored pupil and multilayer response',0,1)
    integrator = DebyeIntegrator(config,(result.pupil_ex,result.pupil_ey,result.pupil_x,result.pupil_y))
    _check_cancel(cancel)
    units = int(axial)+(g.volume_z if volume else 0)
    field_progress = (lambda value:progress(.99*value)) if progress else None
    changes = {'config':config}
    if axial:
        xz,yz = integrator.generate_focal_field_axial(result.x,result.y,result.z,g.x0,g.y0,
            _stage_callback(field_progress,stage_progress,'XZ and YZ sections',len(result.z),0,1/units),cancel=cancel)
        changes.update(xz=xz/result.normalization,yz=yz/result.normalization)
    if volume:
        volume_started = perf_counter()
        vx = np.linspace(g.x0-g.lateral_range/2,g.x0+g.lateral_range/2,g.volume_xy)
        vy = np.linspace(g.y0-g.lateral_range/2,g.y0+g.lateral_range/2,g.volume_xy)
        vz = np.linspace(g.z0-g.axial_range/2,g.z0+g.axial_range/2,g.volume_z)
        fields = integrator.generate_focal_field_volume(vx,vy,vz,
            _stage_callback(field_progress,stage_progress,'XYZ volume',g.volume_z,int(axial)/units,g.volume_z/units),cancel=cancel)
        changes.update(volume=fields,volume_x=vx,volume_y=vy,volume_z=vz,
                       volume_seconds=perf_counter()-volume_started)
    _check_cancel(cancel)
    changes['elapsed_seconds'] = (result.elapsed_seconds or 0)+perf_counter()-started
    if progress: progress(1.)
    return replace(result,**changes)
