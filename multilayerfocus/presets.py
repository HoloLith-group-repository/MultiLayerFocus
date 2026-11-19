"""Objectives and materials."""
import numpy as np
import csv
from pathlib import Path

# Objectives of Table 3 of the MultiLayerFocus paper (nominal manufacturer data)
OBJECTIVES = [
    dict(name='Leica HC PL FLUOTAR 10×/0.30', na=0.30, f=20000., n=1.000),
    dict(name='Olympus UPLFLN 20×/0.50', na=0.50, f=9000., n=1.000),
    dict(name='Leica HC PL FLUOTAR 20×/0.55', na=0.55, f=10000., n=1.000),
    dict(name='ZEISS EC Epiplan-Neofluar 20×/0.65', na=0.65, f=8250., n=1.000),
    dict(name='ZEISS EC Plan-Neofluar 40×/0.75', na=0.75, f=4125., n=1.000),
    dict(name='Olympus/Evident MXPLFLN 50×/0.80', na=0.80, f=3600., n=1.000),
    dict(name='Leica HC PL FLUOTAR 40×/0.85', na=0.85, f=5000., n=1.000),
    dict(name='ZEISS Plan-Apochromat 63×/0.95', na=0.95, f=2620., n=1.000),
    dict(name='Leica HCX PL APO 63×/1.20 W', na=1.20, f=3170., n=1.334),
    dict(name='ZEISS EC Plan-Neofluar 100×/1.30 Oil', na=1.30, f=1650., n=1.518),
]



def _csv_nk(material):
    """Read n and k from the local CSV. Wavelength in µm."""
    path = Path(__file__).with_name('materials_nk.csv')

    with path.open(newline='', encoding='utf-8') as f:
        rows = [
            (float(r['wavelength_nm']) / 1000,
             float(r['n']), float(r['k']))
            for r in csv.DictReader(f)
            if r['material'] == material
        ]

    if not rows:
        raise ValueError(f"No optical data for {material}")

    data = np.array(sorted(rows))
    wavelengths, n_values, k_values = data.T

    def nk(wl_um):
        if not wavelengths[0] <= wl_um <= wavelengths[-1]:
            raise ValueError(f"{material}: wavelength outside data range")

        n = np.interp(wl_um, wavelengths, n_values)
        k = np.interp(wl_um, wavelengths, k_values)
        return complex(n, k)

    return nk



# Each material: callable n(wavelength_um) -> complex, plus a note. Constant values are
# typical figures for visible light and are meant as starting points, not reference data.
MATERIALS = {
    'Custom': None,
    'Air': (_csv_nk('Air'), 'Ciddor 1996 — dry air, 15 °C, 101325 Pa; tabulated n(λ), k = 0'),
    'Water': (_csv_nk('Water'), 'Hale & Querry 1973 — liquid water, 25 °C; tabulated n(λ) and k(λ)'),
    'Immersion Oil': (_csv_nk('ZEISS Immersol 518 N'),'ZEISS Immersol 518 N, 23 °C; refractive index interpolated/extrapolated from manufacturer data; k=0 assumed'),
    'BK7 glass': (_csv_nk('BK7 glass'),'SCHOTT N-BK7, 20 °C; n from Sellmeier data; k estimated from internal transmittance'),
    'Fused silica': (_csv_nk('Fused silica'),'Fused silica (SiO2), 20 °C; n from Malitson 1965; k=0 assumed (absorption neglected)'),
    'PMMA': (_csv_nk('PMMA'),'MicroChem PMMA 950; n from manufacturer dispersion data; k=0 assumed (absorption neglected)'),
    'Silver': (_csv_nk('Silver'),'Silver (Ag); experimental optical constants n and k from Johnson & Christy (1972), room temperature'),
    'Aluminium': (_csv_nk('Aluminium'),'Aluminium (Al); optical constants n and k from Rakic (1995), based on experimental evaporated-film data'),
    'Gold': (_csv_nk('Gold'),'Gold (Au); experimental optical constants n and k from Johnson & Christy (1972), room temperature'),
    'Silicon': (_csv_nk('Silicon'),'Crystalline silicon (c-Si), 20 °C; n and k from Franta et al. (2017)'),
    'Titanium dioxide': (_csv_nk('Titanium dioxide'),'TiO2 thin film, e-beam deposited; n and k from Franta et al. (2015)'),
    'Aluminium oxide': (_csv_nk('Aluminium oxide'),'Al2O3 thin film, e-beam deposited; n and k from Franta et al. (2015)'),
}


POLARIZATIONS = {
    'linear-x': dict(kind='uniform', gamma=0., beta=0.),
    'linear-y': dict(kind='uniform', gamma=90., beta=0.),
    'linear-45°': dict(kind='uniform', gamma=45., beta=0.),
    'circular-L': dict(kind='uniform', gamma=45., beta=90.),
    'circular-R': dict(kind='uniform', gamma=45., beta=-90.),
    'radial': dict(kind='spiral', delta=0.),
    'azimuthal': dict(kind='spiral', delta=90.),
    'custom': None,
}

ZERNIKE_NAMES = ['Piston Z(0,0)', 'Tilt y Z(1,−1)', 'Tilt x Z(1,1)', 'Oblique astig. Z(2,−2)', 'Defocus Z(2,0)',
                 'Vertical astig. Z(2,2)', 'Vertical trefoil Z(3,−3)', 'Vertical coma Z(3,−1)', 'Horizontal coma Z(3,1)',
                 'Oblique trefoil Z(3,3)', 'Oblique quadrafoil Z(4,−4)', 'Oblique 2nd astig. Z(4,−2)',
                 'Primary spherical Z(4,0)', 'Vertical 2nd astig. Z(4,2)', 'Vertical quadrafoil Z(4,4)']
