"""Scene model, automatic sampling, simulation runs and derived metrics (no GUI code)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
import json, math, time
import numpy as np

from stratified_focus import (Objective, Illumination, Stack, Sampling, SimulationConfig, DebyeIntegrator)
from stratified_focus.field import poynting, absorbed_power_density, estimate_sampling_error, Z0
from stratified_focus.multilayer import gff_multilayer_response
from stratified_focus.pupil import generate_inc_field, load_inc_field, polar_nodes, azimuthal_nodes, critical_angles
from stratified_focus.analysis import polarization_ellipse_3d
from stratified_focus.fast import BesselIntegrator, azimuthal_order, fft_size
from . import presets

INF = float('inf')


@dataclass
class Layer:
    name: str = 'Custom'
    n_re: float = 1.0
    n_im: float = 0.0
    d: float | None = None          # thickness in um; None for the two semi-infinite media

    @property
    def n(self):
        return complex(self.n_re, self.n_im)


@dataclass
class Scene:
    # objective & beam
    objective: str = presets.OBJECTIVES[7]['name']
    na: float = 0.95
    focal_length: float = 2620.0
    wavelength: float = 0.532
    polarization: str = 'linear-x'
    gamma: float = 0.0
    beta: float = 0.0
    delta: float = 0.0
    vortex: int = 0
    fill: float = 0.0                # Gaussian waist / pupil radius (0 = uniform illumination)
    zernike: list = field(default_factory=lambda: [0.0]*15)
    power_mW: float = 1.0
    # stack (first = immersion / incident medium)
    layers: list = field(default_factory=lambda: [Layer('Air', 1.0, 0.0, None)])
    z_out: float = 0.0
    # view
    z0: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    lateral: float = 2.0
    axial: float = 3.0
    nxy: int = 121
    nz: int = 121
    # accuracy
    quality: str = 'auto'            # auto | draft | standard | high | custom
    ntheta: int = 64
    nphi: int = 96
    volume: bool = False
    volume_xy: int = 61
    volume_z: int = 61
    pupil_files: list | None = None  # Ex, Ey, x, y text files (measured / designed pupil)
    # depth-aberration pre-compensation by a pupil phase (as displayed on an SLM conjugate to the pupil)
    correct: bool = False
    correct_z: float = 0.0           # target point on the axis where all plane-wave components arrive in phase

    # --------------------------------------------------------------- conversion
    def stack(self):
        n = tuple(l.n for l in self.layers)
        d = tuple(INF if (i in (0, len(self.layers)-1)) else float(l.d) for i, l in enumerate(self.layers))
        if len(n) == 1:
            d = (INF,)
        return Stack(n, d, self.z_out)

    def pupil_radius(self):
        return self.focal_length*self.na/self.layers[0].n_re

    def illumination(self):
        p = presets.POLARIZATIONS.get(self.polarization) or {}
        kind = p.get('kind', 'uniform' if self.polarization != 'custom-cylindrical' else 'spiral')
        gamma = p.get('gamma', self.gamma); beta = p.get('beta', self.beta); delta = p.get('delta', self.delta)
        waist = 6e6*max(1.0, self.pupil_radius()) if self.fill <= 0 else self.fill*self.pupil_radius()
        return Illumination(self.wavelength, waist, kind, gamma, beta, delta, int(self.vortex),
                            tuple(float(v) for v in self.zernike))

    def config(self, ntheta=None, nphi=None, axial=True, volume=None):
        obj = Objective(self.na, self.focal_length, self.layers[0].n_re)
        nt, npf = ntheta or self.ntheta, nphi or self.nphi
        vol = self.volume if volume is None else volume
        s = Sampling(int(npf), int(nt), int(self.nxy), int(self.nz), float(self.lateral), float(self.axial),
                     float(self.x0), float(self.y0), float(self.z0), bool(axial), bool(vol),
                     int(self.volume_xy), int(self.volume_z), quadrature='gauss')
        return SimulationConfig(obj, self.illumination(), self.stack(), s)

    def validate(self):
        if not (0 < self.na < self.layers[0].n_re):
            raise ValueError(f'NA must be below the immersion index n = {self.layers[0].n_re:g}')
        for i, l in enumerate(self.layers[1:-1], start=2):
            if l.d is None or not (l.d >= 0):
                raise ValueError(f'Layer {i}: finite thickness required')
        self.config(8, 9).validate()

    # --------------------------------------------------------------- persistence
    def to_json(self):
        d = asdict(self)
        return json.dumps(dict(format='MultiLayerFocus scene', version=1, scene=d), indent=2)

    @staticmethod
    def from_json(text):
        data = json.loads(text)
        d = data.get('scene', data)
        d['layers'] = [Layer(**l) for l in d['layers']]
        known = Scene.__dataclass_fields__
        return Scene(**{k: v for k, v in d.items() if k in known})


def correction_phase(scene, theta):
    """Pupil phase (rad) that brings every plane-wave component in phase at z = scene.correct_z on the axis.

    It cancels the geometric phase accumulated through the stack (sum of k_z d over the media crossed, with
    the phase reference of the solver: nominal focus z = 0 in the immersion medium). Evanescent components
    carry no propagation phase. This is the phase an SLM conjugate to the pupil should add.
    """
    st = scene.stack(); n = np.asarray(st.indices, complex); N = len(n)
    k0 = 2*np.pi/scene.wavelength
    s1 = np.real(n[0])*np.sin(np.asarray(theta, float))
    kz = [np.real(k0*np.sqrt(nj**2 - s1**2 + 0j)) for nj in n]
    zt = float(scene.correct_z)
    if N == 1:
        path = zt*kz[0]
    else:
        region = st.locate_medium(zt)[0]
        path = st.z_in*kz[0]
        for j in range(1, min(region, N-1)):
            path = path + kz[j]*st.thicknesses[j]
        if region == 0:
            path = zt*kz[0]
        elif region == N-1:
            path = path + (zt - st.z_out)*kz[N-1]
        else:
            path = path + (zt - st.interfaces[region-1])*kz[region]
    return -(path - path[0]) if np.ndim(path) else -path


def pupil_for(scene, cfg):
    """Imported and/or phase-corrected pupil on the solver grid, or None for the plain generated one."""
    if not scene.pupil_files and not scene.correct:
        return None
    g, obj = cfg.sampling, cfg.objective
    bp = critical_angles(cfg.stack, float(np.real(cfg.stack.indices[0])), obj.theta_max)
    if scene.pupil_files:
        ex, ey, xp, yp = load_inc_field(*scene.pupil_files, obj, g.phi, g.theta, 'gauss', bp)
    else:
        ex, ey, xp, yp = generate_inc_field(obj, cfg.illumination, g.phi, g.theta, 'gauss', bp)
    if scene.correct:
        theta, _ = polar_nodes(obj.theta_max, g.theta, 'gauss', bp)
        ph = np.exp(1j*correction_phase(scene, theta))[None, :]
        ex, ey = ex*ph, ey*ph
    return ex, ey, xp, yp


_ORDER_CACHE = {}


def pupil_order(scene):
    """Highest azimuthal order carried by the pupil (decides whether the Bessel solver pays off)."""
    key = json.dumps([scene.polarization, scene.gamma, scene.beta, scene.delta, scene.vortex, scene.fill,
                      list(scene.zernike), scene.pupil_files, scene.na, scene.focal_length, scene.layers[0].n_re,
                      scene.wavelength])
    if key not in _ORDER_CACHE:
        cfg = scene.config(24, 257, axial=False, volume=False)
        if scene.pupil_files:
            ex, ey, _, _ = load_inc_field(*scene.pupil_files, cfg.objective, 257, 24, 'gauss', ())
        else:
            ex, ey, _, _ = generate_inc_field(cfg.objective, cfg.illumination, 257, 24, 'gauss', ())
        _ORDER_CACHE[key] = azimuthal_order(ex, ey)
    return _ORDER_CACHE[key]


def build(scene, ntheta, nphi_2d, want_h=False, axial=True, volume=False, method='auto'):
    """Integrator for the scene: the exact azimuthal (Bessel) solver when the pupil carries few harmonics,
    otherwise the two-dimensional Gauss x periodic-trapezoid quadrature with nphi_2d azimuthal samples."""
    order = pupil_order(scene)
    bessel = method == 'bessel' or (method == 'auto' and 2*order + 1 <= nphi_2d/8)   # few harmonics: Bessel pays off
    nphi = fft_size(order) + 1 if bessel else nphi_2d
    cfg = scene.config(ntheta, nphi, axial=axial, volume=volume)
    cls = BesselIntegrator if bessel else DebyeIntegrator
    return cls(cfg, pupil_for(scene, cfg), want_h=want_h)


def integrator(scene, cfg, want_h=False):
    return DebyeIntegrator(cfg, pupil_for(scene, cfg), want_h=want_h)


QUALITY = {'draft': (24, 48), 'standard': (48, 96), 'high': (96, 192)}
_AUTO_CACHE = {}


def _probe_error(scene, nt, nphi, probe):
    """Relative difference between the solution on (Nt, Nphi) and on (2Nt, 2Nphi-1) at probe points."""
    a = build(scene, nt, nphi, axial=False).evaluate(probe[:, 0], probe[:, 1], probe[:, 2])
    b = build(scene, 2*nt, 2*(nphi-1)+1, axial=False).evaluate(probe[:, 0], probe[:, 1], probe[:, 2])
    sc = np.sqrt(np.max(np.sum(np.abs(b)**2, 0)))
    return float(np.sqrt(np.max(np.sum(np.abs(a-b)**2, 0)))/sc) if sc > 0 else 0.0


def _probe_points(scene):
    xs = np.linspace(scene.x0-scene.lateral/2, scene.x0+scene.lateral/2, 5)
    ys = np.linspace(scene.y0-scene.lateral/2, scene.y0+scene.lateral/2, 5)
    X, Y = np.meshgrid(xs, ys)
    zs = np.linspace(scene.z0-scene.axial/2, scene.z0+scene.axial/2, 15)
    pts = [np.c_[X.ravel(), Y.ravel(), np.full(X.size, scene.z0)],
           np.c_[np.full(zs.size, scene.x0), np.full(zs.size, scene.y0), zs],
           np.c_[np.full(zs.size, scene.x0+scene.lateral/2), np.full(zs.size, scene.y0), zs]]
    return np.vstack(pts)


def auto_sampling(scene: Scene, target='standard', tol=None):
    """Gauss-Legendre node counts: N_phi from the lateral phase budget k0*NA*r_max, N_theta by
    adaptive doubling until the solution on (N, 2N) agrees to `tol` at probe points."""
    tol = tol if tol is not None else {'draft': 3e-2, 'standard': 1e-3, 'high': 1e-5}.get(target, 1e-3)
    key = (json.dumps(asdict(scene), sort_keys=True, default=str), target)
    if key in _AUTO_CACHE:
        return _AUTO_CACHE[key]
    k0 = 2*np.pi/scene.wavelength
    r_max = math.hypot(abs(scene.x0)+scene.lateral/2, abs(scene.y0)+scene.lateral/2)
    lateral_phase = k0*scene.na*r_max
    base_t, base_p = QUALITY.get(target, QUALITY['standard'])
    nphi = int(min(2001, max(base_p, math.ceil(2.2*lateral_phase) + 24)))
    probe = _probe_points(scene)
    nt, err = base_t, None
    while nt <= 1024:
        err = _probe_error(scene, nt, nphi, probe)
        if err < tol:
            break
        nt *= 2
    out = (min(nt, 1024), nphi, dict(lateral_phase=lateral_phase, estimated_error=err))
    _AUTO_CACHE[key] = out
    return out


def sampling_for(scene: Scene, preview=False):
    if preview:
        light = replace(scene, z0=scene.z0)   # copy
        return auto_sampling(light, 'draft')
    if scene.quality == 'custom':
        return scene.ntheta, scene.nphi, {}
    target = 'high' if scene.quality == 'auto' else scene.quality
    return auto_sampling(scene, target)


@dataclass
class Run:
    scene: Scene
    config: SimulationConfig
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray                      # ascending
    Exy: np.ndarray                    # raw complex (3, ny, nx) at z0
    Exz: np.ndarray                    # raw complex (3, nz, nx) at y = y0
    Eyz: np.ndarray                    # raw complex (3, nz, ny) at x = x0
    Hxy: np.ndarray | None = None
    Hxz: np.ndarray | None = None
    volume: np.ndarray | None = None
    vx: np.ndarray | None = None
    vy: np.ndarray | None = None
    vz: np.ndarray | None = None
    scale: float = 1.0                 # raw -> V/m for scene.power_mW
    budget: dict | None = None
    metrics: dict | None = None
    sampling_error: float | None = None
    seconds: float = 0.0
    preview: bool = False
    ntheta: int = 0
    nphi: int = 0
    pupil: tuple | None = None
    angular: object = None
    method: str = ''                   # 'Bessel (N orders)' or '2-D quadrature'
    Exz_ref: np.ndarray | None = None  # uncorrected XZ section when the aberration correction is on
    metrics_ref: dict | None = None
    correction: tuple | None = None    # (theta, phase) of the applied pupil correction


def run(scene: Scene, preview=False, want_h=True, check=False, progress=None, cancel=None, method='auto'):
    t0 = time.perf_counter()
    scene.validate()
    nt, npf, _ = sampling_for(scene, preview)
    vol_on = scene.volume and not preview
    I = build(scene, nt, npf, want_h=want_h, axial=True, volume=vol_on, method=method)
    cfg = I.config
    g = cfg.sampling
    nxy = min(g.xy, 81) if preview else g.xy
    nz = min(g.axial, 101) if preview else g.axial
    x = np.linspace(g.x0-g.lateral_range/2, g.x0+g.lateral_range/2, nxy)
    y = np.linspace(g.y0-g.lateral_range/2, g.y0+g.lateral_range/2, nxy)
    z = np.linspace(g.z0-g.axial_range/2, g.z0+g.axial_range/2, nz)
    def chk():
        if cancel is not None and cancel():
            raise RuntimeError('cancelled')
    Exy = I.generate_focal_field_xy(x, y, g.z0); chk()
    if progress: progress(0.3)
    Exz, Eyz = I.generate_focal_field_axial(x, y, z, g.x0, g.y0); chk()
    if progress: progress(0.55)
    Hxy = Hxz = None
    if want_h:
        Hxy = I.generate_focal_field_xy(x, y, g.z0, which='H')
        Hxz, _ = I.generate_focal_field_axial(x, y[:1], z, g.x0, g.y0, which='H'); chk()
    if progress: progress(0.7)
    vol = vx = vy = vz = None
    if g.compute_volume:
        vx = np.linspace(g.x0-g.lateral_range/2, g.x0+g.lateral_range/2, g.volume_xy)
        vy = np.linspace(g.y0-g.lateral_range/2, g.y0+g.lateral_range/2, g.volume_xy)
        vz = np.linspace(g.z0-g.axial_range/2, g.z0+g.axial_range/2, g.volume_z)
        vol = I.generate_focal_field_volume(vx, vy, vz, progress=(lambda f: progress(0.7+0.25*f)) if progress else None,
                                            cancel=cancel)
    scale = I.absolute_scale(scene.power_mW*1e-3)
    bessel = isinstance(I, BesselIntegrator)
    r = Run(scene, cfg, x, y, z, Exy, Exz, Eyz, Hxy, Hxz, vol, vx, vy, vz, scale, preview=preview,
            ntheta=nt, nphi=(I.nfft if bessel else npf), pupil=(I.ex, I.ey, I.xp, I.yp), angular=I.angular,
            method=f'Bessel, {len(I.orders)} azimuthal orders' if bessel else f'2-D quadrature, {npf} azimuthal nodes')
    r.budget = energy_budget_dense(scene, cfg)
    r.metrics = metrics(r)
    if scene.correct:
        I0 = build(replace(scene, correct=False), nt, npf, axial=True, method=method)
        r.Exz_ref, _ = I0.generate_focal_field_axial(x, y[:1], z, g.x0, g.y0)
        r.metrics_ref = axial_metrics(r.Exz_ref, x, z, scale, cfg.stack, scene.x0)
        th = np.linspace(0, cfg.objective.theta_max, 400)
        r.correction = (th, correction_phase(scene, th))
    if check and not preview:
        r.sampling_error = _probe_error(scene, nt, npf, _probe_points(scene))
    r.seconds = time.perf_counter()-t0
    if progress: progress(1.0)
    return r


def axial_metrics(Exz, x, z, scale, stack, x0=0.0):
    """Peak intensity, its position and axial FWHM of an XZ section (used for the corrected/uncorrected pair)."""
    Ixz = np.sum(np.abs(Exz)**2, 0)*scale**2
    iz, ix = np.unravel_index(np.argmax(Ixz), Ixz.shape)
    j, _ = stack.locate_medium(float(z[iz]))
    I_peak = 0.5*complex(stack.indices[j]).real*Ixz.max()/Z0
    return dict(z_peak=float(z[iz]), I_peak_Wcm2=float(I_peak*1e-4), fwhm_z=fwhm(z, Ixz[:, ix]),
                fwhm_x=fwhm(x, Ixz[iz, :]))


# ------------------------------------------------------------------- derived quantities

class _BudgetGrid:
    """Minimal stand-in exposing the attributes energy_budget() needs, on a dense polar grid."""
    def __init__(self, scene, cfg, ntheta=512):
        obj, light, st = cfg.objective, cfg.illumination, cfg.stack
        nphi = min(cfg.sampling.phi, 257)
        self.config = cfg
        bp = critical_angles(st, float(np.real(st.indices[0])), obj.theta_max)
        self.theta, self.wtheta = polar_nodes(obj.theta_max, ntheta, 'gauss', bp)
        self.phi, self.wphi = azimuthal_nodes(nphi)
        if scene.pupil_files:
            ex, ey, _, _ = load_inc_field(*scene.pupil_files, obj, nphi, ntheta, 'gauss', bp)
        else:
            ex, ey, _, _ = generate_inc_field(obj, light, nphi, ntheta, 'gauss', bp)
        cp, sp = np.cos(self.phi)[:, None], np.sin(self.phi)[:, None]
        self.ep, self.es = ex*cp + ey*sp, -ex*sp + ey*cp
        self.angular = gff_multilayer_response(st, self.theta, light.wavelength)


def energy_budget_dense(scene, cfg):
    return energy_budget(_BudgetGrid(scene, cfg))


def energy_budget(I):
    """Beam-level reflectance, transmittance and absorptance of every layer (infinite plane).

    Uses the plane-wave amplitudes: the z-flux of each angular component at the bottom and top of each
    medium, weighted by its pupil power. Requires real incident index.
    """
    st = I.config.stack; ar = I.angular
    n = np.asarray(st.indices, complex); N = len(n)
    # pupil power weights per (phi, theta) for p and s
    w = (I.wphi[:, None]*I.wtheta[None, :]*np.sin(I.theta)[None, :]*np.cos(I.theta)[None, :])
    Pp = np.sum(np.abs(I.ep)**2*w, axis=0); Ps = np.sum(np.abs(I.es)**2*w, axis=0)
    total = Pp.sum() + Ps.sum()
    if total <= 0 or N == 1:
        return dict(R=0.0, T=1.0, A=[0.0]*N, names=None)
    c1 = np.cos(ar.theta[0])
    out = {}
    def flux(pol, j, where):
        """Normalised S_z of polarisation pol at bottom ('b') or top ('t') of medium j."""
        F = (ar.ap_plus if pol == 'p' else ar.as_plus)[j]
        B = (ar.ap_minus if pol == 'p' else ar.as_minus)[j]
        ct = np.cos(ar.theta[j]); nj = n[j]
        kz = ar.kz[j]
        d = st.thicknesses[j]
        if 0 < j < N-1:
            ph = np.exp(1j*kz*d)
            Fv, Bv = (F, B*ph) if where == 'b' else (F*ph, B)
        else:
            Fv, Bv = F, B
        if pol == 's':   # E_y = F + B ; Z0 H_x = -n cos (F - B)
            Ey = Fv + Bv; Hx = -nj*ct*(Fv - Bv)
            S = -0.5*np.real(Ey*np.conj(Hx))
        else:            # E_x = cos (F - B) ; Z0 H_y = n (F + B)
            Ex = ct*(Fv - Bv); Hy = nj*(Fv + Bv)
            S = 0.5*np.real(Ex*np.conj(Hy))
        return S/(0.5*n[0].real*np.real(c1))
    Rp = np.abs(ar.rp)**2; Rs = np.abs(ar.rs)**2
    R = float((np.sum(Pp*Rp) + np.sum(Ps*Rs))/total)
    T = float((np.sum(Pp*flux('p', N-1, 'b')) + np.sum(Ps*flux('s', N-1, 'b')))/total)
    A = [0.0]
    for j in range(1, N-1):
        a = (np.sum(Pp*(flux('p', j, 'b') - flux('p', j, 't'))) + np.sum(Ps*(flux('s', j, 'b') - flux('s', j, 't'))))/total
        A.append(float(a))
    A.append(0.0)
    return dict(R=R, T=T, A=A)


def fwhm(x, profile):
    p = np.asarray(profile, float)
    if p.max() <= 0: return float('nan')
    i = int(np.argmax(p)); half = p[i]/2
    l = i
    while l > 0 and p[l] > half: l -= 1
    r = i
    while r < len(p)-1 and p[r] > half: r += 1
    if l == 0 and p[0] > half or r == len(p)-1 and p[-1] > half:
        return float('nan')
    xl = x[l] + (half-p[l])*(x[l+1]-x[l])/(p[l+1]-p[l])
    xr = x[r-1] + (half-p[r-1])*(x[r]-x[r-1])/(p[r]-p[r-1])
    return float(xr-xl)


def metrics(r: Run):
    Ixz = np.sum(np.abs(r.Exz)**2, 0)*r.scale**2
    iz, ix = np.unravel_index(np.argmax(Ixz), Ixz.shape)
    Ixy = np.sum(np.abs(r.Exy)**2, 0)
    jy, jx = np.unravel_index(np.argmax(Ixy), Ixy.shape)
    comps = np.sum(np.abs(r.Exy)**2, axis=(1, 2)); frac = comps/comps.sum() if comps.sum() > 0 else comps
    st = r.config.stack
    j_peak, _ = st.locate_medium(float(r.z[iz]))
    n_peak = complex(st.indices[j_peak])
    I_peak = 0.5*n_peak.real*Ixz.max()/Z0          # W/m^2 (plane-wave intensity convention)
    return dict(z_peak=float(r.z[iz]), x_peak=float(r.x[ix]),
                fwhm_x=fwhm(r.x, Ixy[jy, :]), fwhm_y=fwhm(r.y, Ixy[:, jx]), fwhm_z=fwhm(r.z, Ixz[:, ix]),
                frac=frac.tolist(), E_peak_Vm=float(np.sqrt(Ixz.max())), I_peak_Wcm2=float(I_peak*1e-4),
                medium_peak=int(j_peak))


QUANTITIES = ['|E|²', '|Ex|²', '|Ey|²', '|Ez|²', '|E⊥|²', 'Sz (Poynting)', 'Absorbed power', 'arg Ex', 'arg Ey', 'arg Ez']


def quantity(r: Run, plane: str, name: str):
    """Return (values, unit label) of a display quantity on 'xy' or 'xz'."""
    E = r.Exy if plane == 'xy' else r.Exz
    H = r.Hxy if plane == 'xy' else r.Hxz
    s = r.scale
    if name.startswith('arg'):
        c = 'xyz'.index(name[-1])
        return np.angle(E[c]), 'rad'
    if name == 'Sz (Poynting)':
        if H is None: return np.zeros(E.shape[1:]), ''
        return poynting(E*s, H*s)[2]*1e-4, 'W/cm²'
    if name == 'Absorbed power':
        st = r.config.stack
        if plane == 'xy':
            j, _ = st.locate_medium(float(r.config.sampling.z0))
            return absorbed_power_density(E*s, st.indices[j], r.scene.wavelength)*1e-6, 'W/cm³'
        out = np.zeros(E.shape[1:])
        for i, zz in enumerate(r.z):
            j, _ = st.locate_medium(float(zz))
            out[i] = absorbed_power_density(E[:, i:i+1]*s, st.indices[j], r.scene.wavelength)[0]*1e-6
        return out, 'W/cm³'
    if name == '|E⊥|²':
        return np.sum(np.abs(E[:2]*s)**2, 0), 'V²/m²'
    if name == '|E|²':
        return np.sum(np.abs(E*s)**2, 0), 'V²/m²'
    c = 'xyz'.index(name[2])
    return np.abs(E[c]*s)**2, 'V²/m²'
