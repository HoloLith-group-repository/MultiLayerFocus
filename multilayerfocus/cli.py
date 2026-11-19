"""Batch mode: run saved scenes (and parameter sweeps) without the interface.

    MultiLayerFocus --run scene.json [--out fields.npz] [--quality draft|standard|high]
                          [--sweep NAME=FROM:TO:STEPS] [--csv sweep.csv]

NAME is z0, na, wavelength, z_out, fill, vortex, correct_z, power_mW or d<k> (thickness of layer k, counted from
the immersion medium = 0; as in the Sweep tab, the last finite layer grows upwards from a fixed lower interface).
A run writes the fields (.npz) and a summary (_summary.json); a sweep writes one CSV row per value.
"""
import sys, json, copy, argparse, time
from pathlib import Path
import numpy as np
from .engine import Scene, run
from . import presets

def _setter(name):
    if name.startswith('d') and name[1:].isdigit():
        k = int(name[1:])
        def f(s, v):
            if not 0 < k < len(s.layers)-1:
                raise ValueError(f'layer {k} is not a finite layer')
            old = s.layers[k].d; s.layers[k].d = max(0.0, float(v))
            if k == len(s.layers)-2:                       # keep the bottom of the stack fixed, as the Sweep tab does
                s.z_out += s.layers[k].d - old
        return f
    if name == 'correct_z':
        return lambda s, v: (setattr(s, 'correct', True), setattr(s, 'correct_z', float(v)))
    if name == 'wavelength':
        def set_wavelength(s, v):
            s.wavelength = float(v)
            for layer in s.layers:
                entry = presets.MATERIALS.get(layer.name)
                if entry:
                    n = complex(entry[0](s.wavelength))
                    layer.n_re = round(n.real, 6)
                    layer.n_im = round(n.imag, 6)
        return set_wavelength
    if name in ('z0', 'na', 'z_out', 'fill', 'power_mW'):
        return lambda s, v: setattr(s, name, float(v))
    if name == 'vortex':
        return lambda s, v: setattr(s, 'vortex', int(round(float(v))))
    raise ValueError(f'unknown sweep parameter {name!r}')


def _summary(r):
    m, b = r.metrics, r.budget
    out = dict(seconds=r.seconds, method=r.method, polar_nodes=r.ntheta, estimated_error=r.sampling_error,
               peak_E_Vpm=m['E_peak_Vm'], peak_intensity_Wcm2=m['I_peak_Wcm2'], peak_z_um=m['z_peak'],
               fwhm_x_nm=1000*m['fwhm_x'], fwhm_y_nm=1000*m['fwhm_y'], fwhm_z_nm=1000*m['fwhm_z'],
               fraction_Ex=m['frac'][0], fraction_Ey=m['frac'][1], fraction_Ez=m['frac'][2],
               R=b['R'], T=b['T'], absorbed=b['A'])
    if r.metrics_ref:
        out.update(uncorrected_peak_intensity_Wcm2=r.metrics_ref['I_peak_Wcm2'], uncorrected_peak_z_um=r.metrics_ref['z_peak'],
                   uncorrected_fwhm_z_nm=1000*r.metrics_ref['fwhm_z'])
    return out


def _say(*a):
    if sys.stdout is not None:
        print(*a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='MultiLayerFocus', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='scene file (.json) saved from the interface')
    ap.add_argument('--out', help='field data (.npz); default: next to the scene')
    ap.add_argument('--quality', choices=['draft', 'standard', 'high'], help='accuracy target (default: as saved)')
    ap.add_argument('--sweep', help='NAME=FROM:TO:STEPS')
    ap.add_argument('--csv', help='sweep table (.csv); default: next to the scene')
    a = ap.parse_args(argv)
    src = Path(a.run)
    scene = Scene.from_json(src.read_text(encoding='utf-8'))
    if a.quality:
        scene.quality = a.quality
    if a.sweep:
        name, rng = a.sweep.split('=')
        lo, hi, n = rng.split(':')
        values = np.linspace(float(lo), float(hi), int(n))
        setter = _setter(name.strip())
        rows = []
        t0 = time.perf_counter()
        for v in values:
            s = copy.deepcopy(scene); setter(s, v)
            r = run(s, want_h=False, check=False)
            row = dict(value=float(v), **{k: (sum(x) if k == 'absorbed' else x) for k, x in _summary(r).items()})
            rows.append(row); _say(f'{name} = {v:.6g}: peak {row["peak_intensity_Wcm2"]:.4g} [W/cm²] at z = {row["peak_z_um"]:+.3f} [µm]')
        out = Path(a.csv) if a.csv else src.with_name(src.stem + f'_sweep_{name}.csv')
        keys = [k for k in rows[0] if not isinstance(rows[0][k], (list, str, type(None)))]
        np.savetxt(out, np.array([[row[k] for k in keys] for row in rows], float), delimiter=',', fmt='%.8g',
                   header=','.join(keys), comments='')
        _say(f'{len(rows)} runs in {time.perf_counter()-t0:.1f} [s] -> {out}')
        return 0
    r = run(scene, check=True)
    out = Path(a.out) if a.out else src.with_suffix('.npz')
    d = dict(x_um=r.x, y_um=r.y, z_um=r.z, E_xy_Vpm=r.Exy*r.scale, E_xz_Vpm=r.Exz*r.scale, E_yz_Vpm=r.Eyz*r.scale,
             Z0H_xy_Vpm=r.Hxy*r.scale, Z0H_xz_Vpm=r.Hxz*r.scale, scene_json=np.asarray(scene.to_json()))
    if r.volume is not None:
        d.update(E_volume_Vpm=r.volume*r.scale, vx_um=r.vx, vy_um=r.vy, vz_um=r.vz)
    if r.correction is not None:
        d.update(correction_theta_rad=r.correction[0], correction_phase_rad=r.correction[1])
    np.savez_compressed(out, **d)
    summary = _summary(r)
    out.with_name(out.stem + '_summary.json').write_text(json.dumps(summary, indent=1, default=float), encoding='utf-8')
    _say(json.dumps(summary, indent=1, default=float)); _say(f'-> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
