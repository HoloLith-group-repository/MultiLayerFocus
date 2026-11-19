"""Vectorial focusing in stratified media, lengths in micrometres."""
__version__ = "1.2.0"

from .model import Objective, Illumination, Stack, Sampling, SimulationConfig
from .field import (DebyeIntegrator, Result, simulate, compute_additional, SimulationCancelled,
                    poynting, absorbed_power_density, estimate_sampling_error)
from .analysis import intensity, normalize_focal_fields, polarization_ellipse_3d
from .multilayer import (mtm_compute_snell_angles, mtm_fresnel_coefficients,
                         multilayer_transfer_matrix, gff_multilayer_response,
                         stable_multilayer, power_coefficients)
from .pupil import generate_pupil_coordinates, generate_inc_field, load_inc_field
from .fast import BesselIntegrator

__all__ = ['Objective', 'Illumination', 'Stack', 'Sampling', 'SimulationConfig',
           'DebyeIntegrator', 'BesselIntegrator', 'Result', 'simulate', 'compute_additional', 'SimulationCancelled', 'intensity',
           'generate_pupil_coordinates', 'generate_inc_field', 'load_inc_field',
           'mtm_compute_snell_angles', 'mtm_fresnel_coefficients',
           'multilayer_transfer_matrix', 'gff_multilayer_response',
           'normalize_focal_fields', 'polarization_ellipse_3d', 'poynting',
           'absorbed_power_density', 'estimate_sampling_error', 'stable_multilayer', 'power_coefficients']
