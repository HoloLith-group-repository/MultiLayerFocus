"""Export every numeric plot of the selected tab as separate TXT or MAT datasets.

Map files contain the displayed scientific quantity (before color mapping/log
color transformation) and full 2-D coordinate grids.  A line plot contains
one value column and its matching X coordinates; it has no second independent
spatial coordinate.

This module deliberately does not import Qt, VTK, or the scientific solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import numpy as np


@dataclass
class Dataset:
    name: str
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray | None = None
    kind: str = 'map'
    title: str = ''
    x_label: str = ''
    y_label: str = ''
    unit: str = ''
    transform: str = 'linear'
    labels: list[str] = field(default_factory=list)


def safe_name(value: str) -> str:
    """Produce a portable folder and MATLAB variable name."""
    text = unicodedata.normalize('NFKD', str(value))
    text = text.replace('µ', 'u').replace('μ', 'u').replace('ϕ', 'phi')
    text = ''.join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r'[^A-Za-z0-9_]+', '_', text).strip('_')
    text = re.sub('_+', '_', text)
    if not text:
        text = 'dataset'
    if not text[0].isalpha():
        text = 'D_' + text
    return text[:90]


def _axis_label(plot, axis: str) -> str:
    try:
        ax = plot.getAxis(axis)
        label = ax.label
        return f'{label.text} {label.units}'.strip()
    except (AttributeError, TypeError):
        return ''


def map_dataset(name: str, widget) -> Dataset | None:
    """Use original plotted map values, never the color-mapped pixels."""
    if getattr(widget, 'values', None) is None:
        return None
    values = np.asarray(widget.values)
    x, y = np.asarray(widget.x), np.asarray(widget.y)
    if values.ndim != 2 or values.shape != (len(y), len(x)):
        raise ValueError(f'{name}: map data shape does not match its axes')
    return Dataset(
        name=name, values=values.copy(), x=x.copy(), y=y.copy(),
        kind='map', title=getattr(widget, 'export_title', name),
        x_label=_axis_label(widget.plot, 'bottom'),
        y_label=_axis_label(widget.plot, 'left'),
        unit=getattr(widget, 'display_unit', ''),
        transform='log10 color scale (numeric data remain linear)'
                  if getattr(widget, 'display_log', False) else 'linear',
    )

def _axis_unit(plot, axis: str) -> str:
    label = _axis_label(plot, axis)
    match = re.search(r'\[([^\]]+)\]', label)
    return match.group(1).strip() if match else 'dimensionless'

def line_datasets(prefix: str, widget) -> list[Dataset]:
    result = []
    if widget is None:
        return result
    item = widget.getPlotItem()
    for index, curve in enumerate(item.listDataItems(), 1):
        x, values = curve.getData()
        if x is None or values is None:
            continue
        x, values = np.asarray(x), np.asarray(values)
        if not x.size or x.ndim != 1 or values.shape != x.shape:
            continue
        try:
            curve_name = curve.name()
        except (AttributeError, TypeError):
            curve_name = ''
        name = f'{prefix}_{curve_name or f"curve_{index}"}'
        result.append(Dataset(
            name=name, values=values.copy(), x=x.copy(), kind='curve',
            title=curve_name or name,
            x_label=_axis_label(item, 'bottom'),
            y_label=_axis_label(item, 'left'),
            unit=_axis_unit(item, 'left'),
        ))
    return result


def collect_tab(window) -> list[Dataset]:
    """Gather just the plots that belong to the *selected tab*, not zoom area."""
    tab = window.tabs.currentIndex()
    result: list[Dataset] = []

    map_groups = {
        0: [('Ex', window.pup_I), ('Ey', window.pup_Iy),
            ('Ex_phase', window.pup_P), ('Ey_phase', window.pup_Py)],
        2: [('Focal_axial', window.map_xz), ('Focal_XY', window.map_xy)],
        3: list(zip(('Ex', 'Ey', 'Ez', 'Ex_phase', 'Ey_phase', 'Ez_phase'), window.comp_maps)),
        4: [('Polarization_ellipses_background', window.ell_map),
            ('Ellipticity', window.pol_maps[0]), ('Spin_density', window.pol_maps[1])],
        5: [('Poynting_XZ', window.e_sxz), ('Absorption_XZ', window.e_abs)],
        6: [('Volume_XY_slice', window.vol_maps[0]),
            ('Volume_XY_max_projection', window.vol_maps[1]),
            ('Volume_axial_projection', window.vol_maps[2])],
    }
    for name, widget in map_groups.get(tab, []):
        dataset = map_dataset(name, widget)
        if dataset is not None:
            result.append(dataset)

    curve_groups = {
        1: [('Power_coefficients', window.resp_R),
            ('Fresnel_amplitudes', window.resp_amp)],
        2: [('Axial_profile', window.prof_ax),
            ('Lateral_profile', window.prof_lat)],
        7: [('Parameter_sweep', window.sw_plot)],
    }
    for prefix, widget in curve_groups.get(tab, []):
        result.extend(line_datasets(prefix, widget))

    if tab == 4:
        # The ellipse outlines are graphical curves over the intensity map.
        # Keep them in a single dataset, separated by NaNs to preserve paths.
        x_coords, y_coords = [], []
        for ellipse in getattr(window, 'ellipse_items', []):
            x, y = ellipse.getData()
            if x is None or y is None:
                continue
            x_coords.extend(np.asarray(x, dtype=float).tolist() + [np.nan])
            y_coords.extend(np.asarray(y, dtype=float).tolist() + [np.nan])
        if x_coords:
            result.append(Dataset(
                name='Polarization_ellipse_outlines',
                x=np.asarray(x_coords), values=np.asarray(y_coords),
                kind='curve', title='Polarization ellipse outline coordinates',
                x_label='x [µm]', y_label='y [µm]', unit='µm',
            ))

    if tab == 5 and window.run_result is not None:
        r = window.run_result
        b = r.budget
        names = [layer.name.split(' (')[0] for layer in r.scene.layers]
        parts = [('reflected', b['R'])] + [
            (f'absorbed in {names[j]}', a)
            for j, a in enumerate(b['A']) if a > 1e-6
        ] + [('transmitted', b['T'])]
        fraction = np.asarray([v for _, v in parts], dtype=float)
        centers = np.cumsum(fraction) - fraction / 2
        result.append(Dataset(
            name='Energy_budget', values=fraction, x=centers, kind='curve',
            title='Power budget of the focused beam',
            x_label='stacked bar center (fraction)', y_label='power fraction',
            unit='fraction', labels=[label for label, _ in parts],
        ))

    return result


def _unique_destination(parent: Path, basename: str) -> Path:
    target = parent / basename
    index = 2
    while target.exists():
        target = parent / f'{basename}_{index}'
        index += 1
    return target


def _write_array(path: Path, array: np.ndarray, fmt: str, variable: str):
    if fmt == 'txt':
        np.savetxt(path, array, delimiter='\t', fmt='%.17g')
    else:
        from scipy.io import savemat
        savemat(str(path), {variable: array}, do_compression=True, oned_as='column')


def write_tab(parent: str | Path, tab_title: str, datasets: list[Dataset],
              fmt: str, metadata: dict, volume=None, folder_name=None) -> Path:
    """Create one new folder for the tab with independent dataset folders.

    Each map folder has DATA.ext, x.ext, y.ext; x/y are 2-D matrices
    aligned element-by-element with DATA.  Each curve folder has DATA.ext
    and x.ext (both N x 1): its dependent y values *are* DATA.
    """
    if fmt not in ('txt', 'mat'):
        raise ValueError('Format must be txt or mat')
    if not datasets:
        raise ValueError('There are no numeric plots in the selected tab')

    parent = Path(parent)
    if not parent.is_dir():
        raise FileNotFoundError(f'Destination directory does not exist: {parent}')
    target = _unique_destination(parent, folder_name or f'{safe_name(tab_title)}_{fmt.upper()}')
    target.mkdir()
    names_used = set()
    catalog = []
    try:
        for dataset in datasets:
            stem = safe_name(dataset.name)
            root_stem = stem
            i = 2
            while stem.lower() in names_used:
                stem = f'{root_stem}_{i}'
                i += 1
            names_used.add(stem.lower())
            folder = target / stem
            folder.mkdir()

            values = np.asarray(dataset.values)
            if dataset.kind == 'map':
                if dataset.y is None or values.shape != (len(dataset.y), len(dataset.x)):
                    raise ValueError(f'{stem}: inconsistent map grid')
                xx, yy = np.meshgrid(np.asarray(dataset.x), np.asarray(dataset.y))
            elif dataset.kind == 'curve':
                if values.shape != np.asarray(dataset.x).shape:
                    raise ValueError(f'{stem}: inconsistent curve coordinates')
                values = values.reshape(-1, 1)
                xx, yy = np.asarray(dataset.x).reshape(-1, 1), None
            else:
                raise ValueError(f'{stem}: unsupported dataset kind')

            _write_array(folder / f'{stem}.{fmt}', values, fmt, stem)
            _write_array(folder / f'x.{fmt}', xx, fmt, 'x')
            if yy is not None:
                _write_array(folder / f'y.{fmt}', yy, fmt, 'y')
            if dataset.labels:
                (folder / 'labels.txt').write_text('\n'.join(dataset.labels) + '\n', encoding='utf-8')

            catalog.append({
                'folder': stem, 'original_name': dataset.name,
                'title': dataset.title, 'type': dataset.kind,
                'array_shape': list(values.shape), 'x_axis': dataset.x_label,
                'y_axis': dataset.y_label, 'value_unit': dataset.unit,
                'visual_transform': dataset.transform,
                'data_filename': f'{stem}.{fmt}',
                'x_filename': f'x.{fmt}',
                'y_filename': f'y.{fmt}' if yy is not None else None,
                'labels_filename': 'labels.txt' if dataset.labels else None,
            })
        slice_info = None
        if volume is not None:
            field, xs, ys, zs = volume
            V = np.sum(np.abs(field)**2, axis=0)
            if V.shape != (len(zs), len(ys), len(xs)):
                raise ValueError('Volume dimensions do not match XYZ coordinates')
            peak = V.max() or 1
            slices_dir = target / 'slices'
            slices_dir.mkdir()
            xx, yy = np.meshgrid(xs, ys)
            _write_array(slices_dir / f'x.{fmt}', xx, fmt, 'x')
            _write_array(slices_dir / f'y.{fmt}', yy, fmt, 'y')
            for iz, z in enumerate(zs):
                name = f'slice_z={float(z):.12g}'
                if (slices_dir / f'{name}.{fmt}').exists():
                    name += f'_{iz}'
                _write_array(slices_dir / f'{name}.{fmt}', V[iz] / peak, fmt, 'slice')
            slice_info = {'folder': 'slices', 'count': len(zs), 'quantity': 'normalized total intensity', 'shape': [len(ys), len(xs)], 'z_unit': 'um', 'coordinates': ['x', 'y']}
        record = dict(metadata)
        record.update({
            'application': 'MultiLayerFocus',
            'exported_at_utc': datetime.now(timezone.utc).isoformat(),
            'tab_name': tab_title,
            'format': fmt,
            'data_convention': (
                'MAP: data[row, col], x[row, col], y[row, col] are aligned; '
                'CURVE: data[:, 0] is the dependent Y value and x[:, 0] its X coordinate. '
                'TXT files are tab-separated, unlabelled numeric matrices. '
                'MAT files contain one variable per file. '
                'Numeric map data are before any log-color transformation.'
            ),
            'datasets': catalog,
        })
        if slice_info is not None:
            record['volume_slices'] = slice_info
        (target / 'metadata.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )
        return target
    except Exception:
        # Do not leave a partly generated export after errors.
        import shutil
        shutil.rmtree(target, ignore_errors=True)
        raise
