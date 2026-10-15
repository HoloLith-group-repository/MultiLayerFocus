"""Vector Debye integration across planar stacks."""
from dataclasses import dataclass, replace
from time import perf_counter
from collections.abc import Callable
import numpy as np
from .model import SimulationConfig
from .pupil import generate_inc_field, generate_pupil_coordinates
from .multilayer import gff_multilayer_response, AngularResponse
from .analysis import normalize_focal_fields, polarization_ellipse_3d, Polarization


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


class DebyeIntegrator:
    """Precompute pupil and stack terms, then integrate arbitrary point sets.

    Arrays have phi as their first angular axis and theta as the second.
    At the first interface the incident side is selected; at the last,
    the exit side is selected.
    """
    def __init__(self, config, pupil=None):
        config.validate()
        self.config = config
        obj, light, stack, grid = config.objective, config.illumination, config.stack, config.sampling
        self.ex, self.ey, self.xp, self.yp = pupil if pupil is not None else generate_inc_field(obj, light, grid.phi, grid.theta)
        if self.ex.shape != (grid.phi, grid.theta) or self.ey.shape != self.ex.shape:
            raise ValueError('Pupil fields must have shape (phi samples, theta samples)')
        self.ex, self.ey = np.asarray(self.ex, complex), np.asarray(self.ey, complex)
        self.k1 = 2*np.pi*stack.indices[0]/light.wavelength
        self.f = obj.focal_length
        theta, phi, _, _ = generate_pupil_coordinates(obj, grid.phi, grid.theta)
        self.angular = gff_multilayer_response(stack, theta, light.wavelength)
        th, ph = np.meshgrid(theta, phi)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        wtheta = np.full(grid.theta, obj.theta_max/(grid.theta-1)); wtheta[[0,-1]] *= 0.5
        wphi = np.full(grid.phi, 2*np.pi/(grid.phi-1)); wphi[[0,-1]] *= 0.5
        weight = np.sqrt(ct)*st*wphi[:, None]*wtheta[None, :]
        ep = self.ex*cp + self.ey*sp
        es = -self.ex*sp + self.ey*cp
        ar = self.angular
        ctrans, strans = np.cos(ar.theta[-1]), np.sin(ar.theta[-1])
        def flat(*args):
            return np.stack(args, axis=0).reshape(3, -1)
        self.v_i = flat(-es*sp+ep*ct*cp, es*cp+ep*ct*sp, -ep*st)*weight.ravel()
        self.v_r = flat(-ar.rs*es*sp-ar.rp*ep*ct*cp,
                        ar.rs*es*cp-ar.rp*ep*ct*sp, -ar.rp*ep*st)*weight.ravel()
        self.v_t = flat(-ar.ts*es*sp+ar.tp*ep*ctrans*cp,
                        ar.ts*es*cp+ar.tp*ep*ctrans*sp, -ar.tp*ep*strans)*weight.ravel()
        self.ep, self.es, self.weight = ep, es, weight
        self.sp, self.cp = sp, cp
        self.st, self.ct = st, ct
        self.lx = (self.k1*st*cp).ravel()
        self.ly = (self.k1*st*sp).ravel()
        self.kz_i = (self.k1*ct).ravel()
        self.kz_t = np.broadcast_to(ar.kz[-1], st.shape).ravel()
        self.prefactor = -1j*self.k1*self.f*np.exp(-1j*self.k1*self.f)/(2*np.pi)
        self._internal_cache = {}

    def _internal_vectors(self, medium):
        if medium not in self._internal_cache:
            ar = self.angular
            cp, sp = self.cp, self.sp
            ct, st = np.cos(ar.theta[medium]), np.sin(ar.theta[medium])
            pp = self.ep*ar.ap_plus[medium]
            pm = self.ep*ar.ap_minus[medium]
            ss = self.es*ar.as_plus[medium]
            sm = self.es*ar.as_minus[medium]
            plus = np.stack((-sp*ss+cp*ct*pp, cp*ss+sp*ct*pp, -st*pp)).reshape(3,-1)*self.weight.ravel()
            minus = np.stack((-sp*sm-cp*ct*pm, cp*sm-sp*ct*pm, -st*pm)).reshape(3,-1)*self.weight.ravel()
            kz = np.broadcast_to(ar.kz[medium], self.st.shape).ravel()
            self._internal_cache[medium] = plus, minus, kz
        return self._internal_cache[medium]

    def evaluate(self, x, y, z, chunk=64):
        """Pointwise complex (Ex,Ey,Ez), with broadcastable x/y/z arrays."""
        stack = self.config.stack
        xx, yy, zz = np.broadcast_arrays(x, y, z)
        output = np.empty((3, xx.size), complex)
        xf, yf, zf = xx.ravel(), yy.ravel(), zz.ravel()
        # Group by medium to reuse layer data; chunk angular broadcast for bounded memory.
        if np.all(zf == zf[0]):
            medium = np.full(len(zf), stack.locate_medium(float(zf[0]))[0], dtype=int)
        else:
            interfaces = stack.interfaces
            medium = np.searchsorted(interfaces, zf, side='right')
            medium[zf <= stack.z_in] = 0
            medium[zf >= stack.z_out] = len(stack.indices)-1
        for region in np.unique(medium):
            ids = np.flatnonzero(medium == region)
            for start in range(0, len(ids), chunk):
                sel = ids[start:start+chunk]
                lateral = xf[sel, None]*self.lx + yf[sel, None]*self.ly
                zlocal = zf[sel, None]
                if region == 0:
                    pi = np.exp(1j*(lateral+zlocal*self.kz_i))
                    pr = np.exp(1j*(lateral+(2*stack.z_in-zlocal)*self.kz_i))
                    value = self.v_i@pi.T + self.v_r@pr.T
                elif region == len(stack.indices)-1:
                    pt = np.exp(1j*(lateral+stack.z_in*self.kz_i+(zlocal-stack.z_out)*self.kz_t))
                    value = self.v_t@pt.T
                else:
                    vp, vm, kz = self._internal_vectors(region)
                    zeta = zlocal-stack.interfaces[region-1]
                    common = lateral+stack.z_in*self.kz_i
                    pp = np.exp(1j*(common+zeta*kz))
                    pm = np.exp(1j*(common-zeta*kz))
                    value = vp@pp.T + vm@pm.T
                output[:, sel] = self.prefactor*value
        return output.reshape((3,)+xx.shape)

    def _angular_coefficients(self, z):
        """Weighted angular vector coefficients at a single axial coordinate."""
        stack = self.config.stack
        region, zeta = stack.locate_medium(float(z))
        if region == 0:
            return (self.v_i*np.exp(1j*z*self.kz_i)[None, :] +
                    self.v_r*np.exp(1j*(2*stack.z_in-z)*self.kz_i)[None, :])
        if region == len(stack.indices)-1:
            return self.v_t*np.exp(1j*(stack.z_in*self.kz_i+
                                       (z-stack.z_out)*self.kz_t))[None, :]
        vp, vm, kz = self._internal_vectors(region)
        base = stack.z_in*self.kz_i
        return (vp*np.exp(1j*(base+zeta*kz))[None, :] +
                vm*np.exp(1j*(base-zeta*kz))[None, :])

    def generate_focal_field_xy(self, x, y, z, progress=None, cancel=None):
        """Integrate a regular XY plane using reusable lateral phase matrices."""
        x, y = np.asarray(x), np.asarray(y)
        np_phi, np_theta = self.ex.shape
        coefficients = self._angular_coefficients(z).reshape(3,np_phi,np_theta)
        lx = self.lx.reshape(np_phi,np_theta)
        ly = self.ly.reshape(np_phi,np_theta)
        field = np.zeros((3,len(y),len(x)),complex)
        for t in range(np_theta):
            _check_cancel(cancel)
            x_phase = np.exp(1j*np.outer(lx[:,t],x))
            y_phase = np.exp(1j*np.outer(y,ly[:,t]))
            for component in range(3):
                _check_cancel(cancel)
                field[component] += y_phase @ (coefficients[component,:,t,None]*x_phase)
            if progress:
                progress((t+1)/np_theta)
        _check_cancel(cancel)
        return self.prefactor*field

    def generate_focal_field_volume(self, x, y, z, progress=None, phase_cache_bytes=128*1024**2, cancel=None):
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
            coefficients = self._angular_coefficients(float(zi)).reshape(3, nphi, ntheta)
            plane = fields[:, q]; plane.fill(0)
            for t in range(ntheta):
                _check_cancel(cancel)
                xp, yp = phases[t] if phases is not None else (
                    np.exp(1j*np.outer(lx[:,t], x)), np.exp(1j*np.outer(y, ly[:,t])))
                for c in range(3):
                    _check_cancel(cancel)
                    plane[c] += yp @ (coefficients[c,:,t,None]*xp)
            plane *= self.prefactor
            if progress:
                progress((q+1)/len(z))
        _check_cancel(cancel)
        return fields

    def generate_focal_field_axial(self, x, y, z, x0, y0, progress=None, chunk=8, cancel=None):
        """Integrate XZ and YZ planes while reusing lateral phase matrices."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        x_phase = np.exp(1j*(self.lx[:,None]*x[None,:] + self.ly[:,None]*y0))
        y_phase = np.exp(1j*(self.lx[:,None]*x0 + self.ly[:,None]*y[None,:]))
        xz = np.empty((3,len(z),len(x)),complex)
        yz = np.empty((3,len(z),len(y)),complex)
        for start in range(0,len(z),chunk):
            _check_cancel(cancel)
            stop = min(start+chunk,len(z))
            coefficients = np.stack([self._angular_coefficients(zi) for zi in z[start:stop]])
            xz[:,start:stop] = np.moveaxis(coefficients @ x_phase,0,1)*self.prefactor
            _check_cancel(cancel)
            yz[:,start:stop] = np.moveaxis(coefficients @ y_phase,0,1)*self.prefactor
            if progress:
                progress(stop/len(z))
        _check_cancel(cancel)
        return xz, yz

def simulate(config: SimulationConfig, pupil=None, progress: Callable[[float], None] | None=None,
             stage_progress=None, cancel=None):
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
    _check_cancel(cancel)
    if stage_progress: stage_progress('Normalization and polarization',1,1)
    if progress: progress(1.)
    return Result(config,integrator.ex,integrator.ey,integrator.xp,integrator.yp,
                  integrator.angular,x,y,z,xy,xz,yz,factor,polarization,
                  perf_counter()-started,volume,vx,vy,vz,volume_seconds)


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
