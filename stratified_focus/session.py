"""Portable, versioned NumPy simulation snapshots and legacy NPZ reader."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import tempfile
import numpy as np

from . import __version__
from .analysis import polarization_ellipse_3d
from .field import Result
from .model import Objective, Illumination, Stack, Sampling, SimulationConfig
from .multilayer import gff_multilayer_response

SCHEMA_VERSION = 2
REQUIRED_ARRAYS = ('x', 'y', 'z', 'xy', 'pupil_ex', 'pupil_ey', 'pupil_x', 'pupil_y', 'normalization')


@dataclass
class LoadedSimulation:
    result: Result
    metadata: dict
    legacy: bool = False


def _metadata(result, source_files=()):
    config = asdict(result.config)
    config['stack']['indices'] = [[complex(n).real, complex(n).imag]
                                   for n in result.config.stack.indices]
    config['stack']['thicknesses'] = [None if np.isinf(d) else d
                                      for d in result.config.stack.thicknesses]
    return {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'software': f'MultiLayerFocus {__version__}',
        'units': {'length': 'um', 'phase': 'rad', 'field': 'relative'},
        'config': config,
        'source_files': [Path(path).name for path in source_files],
        'normalization': float(result.normalization),
        'runtime_seconds': result.elapsed_seconds,
        'volume_runtime_seconds': result.volume_seconds,
        'volume_field_convention': 'raw complex fields, axes (component,z,y,x), ascending z',
    }


def save_result(path, result: Result, source_files=()):
    """Save all data needed to replot without solving the optical problem again."""
    path = Path(path)
    metadata = _metadata(result, source_files)
    arrays = dict(x=result.x, y=result.y, z=result.z, xy=result.xy,
                  pupil_ex=result.pupil_ex, pupil_ey=result.pupil_ey,
                  pupil_x=result.pupil_x, pupil_y=result.pupil_y,
                  normalization=np.asarray(result.normalization),
                  metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    if result.xz is not None:
        arrays.update(xz=result.xz, yz=result.yz)
    if result.volume is not None:
        arrays.update(volume=result.volume, volume_x=result.volume_x,
                      volume_y=result.volume_y, volume_z=result.volume_z)
    # Atomic replacement protects an existing snapshot if writing fails.
    handle, temporary = tempfile.mkstemp(suffix='.npz', prefix='.focus-', dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def _config_from_json(config):
    obj = Objective(**config['objective'])
    light_data = dict(config['illumination'])
    light_data['zernike'] = tuple(light_data['zernike'])
    light = Illumination(**light_data)
    st = config['stack']
    indices = tuple(complex(*pair) for pair in st['indices'])
    thicknesses = tuple(np.inf if d is None else float(d) for d in st['thicknesses'])
    stack = Stack(indices, thicknesses, st['z_out'])
    sampling = Sampling(**config['sampling'])
    result = SimulationConfig(obj, light, stack, sampling)
    result.validate()
    return result


def _legacy_config(data):
    """Only the geometry needed by plots is inferred; other inputs remain unknown."""
    indices = tuple(data['indices'].astype(complex))
    thicknesses = tuple(data['thicknesses'].astype(float))
    x, y, z, ex = (data[key] for key in ('x', 'y', 'z', 'pupil_ex'))
    na = float(data['na'])
    sampling = Sampling(phi=ex.shape[0], theta=ex.shape[1], xy=len(x), axial=len(z),
                        lateral_range=float(x[-1]-x[0]), axial_range=float(z[0]-z[-1]),
                        x0=float((x[0]+x[-1])/2), y0=float((y[0]+y[-1])/2),
                        z0=float((z[0]+z[-1])/2), compute_axial='xz' in data.files)
    return SimulationConfig(Objective(na=na, immersion_index=float(indices[0].real)),
                            Illumination(wavelength=float(data['wavelength'])),
                            Stack(indices, thicknesses), sampling)


def _check_arrays(arrays):
    x, y, z, xy, ex, ey, xp, yp = (arrays[key] for key in
                                 ('x','y','z','xy','pupil_ex','pupil_ey','pupil_x','pupil_y'))
    if any(v.ndim != 1 or len(v) < 2 for v in (x, y, z)):
        raise ValueError('Coordinate arrays must be one-dimensional with at least two samples')
    if xy.shape != (3,len(y),len(x)) or ex.ndim != 2 or any(v.shape != ex.shape for v in (ey,xp,yp)):
        raise ValueError('Field or pupil arrays have inconsistent shapes')
    if 'xz' in arrays or 'yz' in arrays:
        if 'xz' not in arrays or 'yz' not in arrays or arrays['xz'].shape != (3,len(z),len(x)) or arrays['yz'].shape != (3,len(z),len(y)):
            raise ValueError('Axial arrays have inconsistent shapes')
    if not np.issubdtype(xy.dtype, np.number) or not np.isfinite(float(arrays['normalization'])):
        raise ValueError('Invalid field values or normalization')
    volume_keys = {'volume', 'volume_x', 'volume_y', 'volume_z'}
    if volume_keys.intersection(arrays):
        if not volume_keys.issubset(arrays):
            raise ValueError('Incomplete volume arrays')
        vx, vy, vz = (arrays[k] for k in ('volume_x', 'volume_y', 'volume_z'))
        for v in (vx, vy, vz):
            if v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)) or not np.all(np.diff(v) > 0):
                raise ValueError('Volume coordinates must be finite and increasing')
            if not np.allclose(np.diff(v), np.diff(v)[0]):
                raise ValueError('Volume coordinates must be uniformly sampled')
        if arrays['volume'].shape != (3,len(vz),len(vy),len(vx)) or arrays['volume'].dtype.kind != 'c':
            raise ValueError('Volume field has an inconsistent shape or dtype')


def load_result(path):
    """Read a snapshot without pickle; legacy GUI exports remain plottable."""
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED_ARRAYS) - set(data.files)
        if missing:
            raise ValueError(f'Missing NPZ arrays: {", ".join(sorted(missing))}')
        keys = list(REQUIRED_ARRAYS) + [key for key in ('xz','yz','volume','volume_x','volume_y','volume_z') if key in data.files]
        arrays = {key: data[key] for key in keys}
        _check_arrays(arrays)
        legacy = 'metadata' not in data.files
        if legacy:
            for key in ('indices','thicknesses','wavelength','na'):
                if key not in data.files:
                    raise ValueError(f'Legacy NPZ lacks {key}; cannot reconstruct the plots')
            cfg = _legacy_config(data)
            metadata = {'schema_version': 0, 'created_at': 'Unknown',
                        'software': 'Earlier NPZ export',
                        'units': {'length': 'um', 'phase': 'rad', 'field': 'relative'},
                        'config': None, 'source_files': [],
                        'normalization': float(arrays['normalization']),
                        'note': 'Full input parameters were not stored in this file.'}
        else:
            raw = data['metadata']
            if raw.ndim != 0 or raw.dtype.kind not in 'US' or raw.itemsize > 1_000_000:
                raise ValueError('Invalid simulation metadata')
            metadata = json.loads(str(raw.item()))
            if metadata.get('schema_version') not in (1, SCHEMA_VERSION):
                raise ValueError(f'Unsupported snapshot schema: {metadata.get("schema_version")}')
            cfg = _config_from_json(metadata['config'])
            g = cfg.sampling
            if (arrays['xy'].shape[1:] != (g.xy,g.xy) or
                    arrays['pupil_ex'].shape != (g.phi,g.theta) or
                    ('xz' in arrays) != g.compute_axial or
                    ('volume' in arrays) != g.compute_volume):
                raise ValueError('Saved array shapes do not match the configuration')
            if g.compute_volume and arrays['volume'].shape != (3,g.volume_z,g.volume_xy,g.volume_xy):
                raise ValueError('Saved volume shape does not match the configuration')
    angles = np.linspace(0, cfg.objective.theta_max, cfg.sampling.theta)
    angular = gff_multilayer_response(cfg.stack, angles, cfg.illumination.wavelength)
    result = Result(cfg, arrays['pupil_ex'], arrays['pupil_ey'], arrays['pupil_x'],
                    arrays['pupil_y'], angular, arrays['x'], arrays['y'], arrays['z'],
                    arrays['xy'], arrays.get('xz'), arrays.get('yz'),
                    float(arrays['normalization']), polarization_ellipse_3d(arrays['xy']),
                    metadata.get('runtime_seconds'), arrays.get('volume'), arrays.get('volume_x'),
                    arrays.get('volume_y'), arrays.get('volume_z'), metadata.get('volume_runtime_seconds'))
    return LoadedSimulation(result, metadata, legacy)
