"""Volume display normalization and streaming raw-data export."""
from pathlib import Path
import numpy as np


def volume_peak(field):
    """Return one common total-intensity peak without allocating a full intensity volume."""
    peak = 0.0
    for q in range(field.shape[1]):
        value = np.sum(np.abs(field[:, q])**2, axis=0)
        if not np.all(np.isfinite(value)):
            raise ValueError('The volume contains invalid intensities')
        peak = max(peak, float(value.max()))
    if peak <= 0:
        raise ValueError('The volume has zero intensity')
    return peak


def volume_intensity(field, component=None, peak=None):
    """Return float32 display intensities using the common total peak for every component."""
    if peak is None:
        peak = volume_peak(field)
    values = np.zeros(field.shape[1:], np.float32)
    for c in range(3) if component is None else (component,):
        # Normalize before casting to avoid overflow for large raw field amplitudes.
        for q in range(field.shape[1]):
            values[q] += (np.abs(field[c,q])**2/peak).astype(np.float32)
    return values


def export_volume_data(folder, result, progress=None):
    """Stream raw complex fields and intensities as coordinate tables, one z plane at a time."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    peak = volume_peak(result.volume)
    xx, yy = np.meshgrid(result.volume_x, result.volume_y)
    filenames = ('ex_intensity', 'ey_intensity', 'ez_intensity', 'total_intensity')
    from contextlib import ExitStack
    with ExitStack() as handles:
        raw = handles.enter_context((folder/'complex_fields.txt').open('w', encoding='utf-8'))
        raw.write('# Raw complex electric fields before XY integral normalization\n'
                  '# columns: x_um y_um z_um Re_Ex Im_Ex Re_Ey Im_Ey Re_Ez Im_Ez\n')
        files = [handles.enter_context((folder/f'{name}.txt').open('w', encoding='utf-8')) for name in filenames]
        for f in files:
            f.write(f'# Common total-intensity peak: {peak:.17g}\n'
                    '# columns: x_um y_um z_um raw_intensity display_intensity\n')
        for q, z in enumerate(result.volume_z):
            coords = (xx.ravel(), yy.ravel(), np.full(xx.size, z))
            fields = result.volume[:,q].reshape(3,-1)
            np.savetxt(raw, np.column_stack((*coords, *[v for field in fields for v in (field.real, field.imag)])), fmt='%.17g')
            powers = np.abs(fields)**2
            for f, values in zip(files, (*powers, powers.sum(axis=0))):
                np.savetxt(f, np.column_stack((*coords, values, values/peak)), fmt='%.17g')
            if progress:
                progress((q+1)/len(result.volume_z))
    return peak
