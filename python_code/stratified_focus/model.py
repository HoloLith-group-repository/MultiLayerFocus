"""Input models. All lengths use micrometres; indices may be complex."""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Objective:
    na: float = 0.5
    focal_length: float = 9000.0
    immersion_index: float = 1.0

    @property
    def theta_max(self):
        return float(np.arcsin(self.na / self.immersion_index))

    def validate(self):
        if not (0 < self.na < self.immersion_index and self.focal_length > 0):
            raise ValueError('Require 0 < NA < immersion index and positive focal length')


@dataclass(frozen=True)
class Illumination:
    wavelength: float = 0.532
    waist: float = 6_000_000.0
    polarization: str = 'uniform'
    gamma_deg: float = 0.0
    beta_deg: float = 0.0
    delta_deg: float = 0.0
    vortex_charge: int = 0
    zernike: tuple[float, ...] = (0.0,) * 15  # Unnormalised real modes through radial degree four; phase in radians.

    def validate(self):
        if self.wavelength <= 0 or self.waist <= 0:
            raise ValueError('Wavelength and pupil waist must be positive')
        if self.polarization not in ('uniform', 'spiral') or len(self.zernike) != 15:
            raise ValueError('Polarization must be uniform or spiral; provide 15 Zernike coefficients')


@dataclass(frozen=True)
class Stack:
    indices: tuple[complex, ...] = (1., 1., 1.)
    thicknesses: tuple[float, ...] = (float('inf'), 0.5, float('inf'))
    z_out: float = 0.0

    def validate(self):
        n, d = self.indices, self.thicknesses
        if len(n) < 1 or len(n) != len(d) or not all(np.isfinite(v) and abs(v) > 0 for v in n):
            raise ValueError('Stack needs at least one finite nonzero refractive index and matching thickness')
        if not (np.isinf(d[0]) and np.isinf(d[-1])) or any(not np.isfinite(v) or v < 0 for v in d[1:-1]):
            raise ValueError('First and last media must be infinite; interior thicknesses finite and nonnegative')
        if not np.isfinite(self.z_out):
            raise ValueError('z_out must be finite')

    @property
    def z_in(self):
        return self.z_out - sum(self.thicknesses[1:-1])

    @property
    def interfaces(self):
        return np.r_[self.z_in, self.z_in + np.cumsum(self.thicknesses[1:-1])]

    def locate_medium(self, z):
        """Return the medium index and local axial coordinate at z."""
        if z >= self.z_out:
            return len(self.indices) - 1, 0.0
        if z <= self.z_in:
            return 0, 0.0
        j = int(np.searchsorted(self.interfaces, z, side='right'))
        return j, z - self.interfaces[j-1]



@dataclass(frozen=True)
class Sampling:
    phi: int = 64
    theta: int = 48
    xy: int = 81
    axial: int = 81
    lateral_range: float = 2.0
    axial_range: float = 2.0
    x0: float = 0.0
    y0: float = 0.0
    z0: float = -0.01
    compute_axial: bool = False
    compute_volume: bool = False
    volume_xy: int = 41
    volume_z: int = 41

    def validate(self):
        if min(self.phi, self.theta, self.xy, self.axial) < 2 or min(self.lateral_range, self.axial_range) <= 0:
            raise ValueError('Sample counts must be at least 2 and ranges positive')
        if min(self.volume_xy, self.volume_z) < 2:
            raise ValueError('Volume sample counts must be at least 2')
        if not all(np.isfinite(v) for v in (self.x0, self.y0, self.z0, self.lateral_range, self.axial_range)):
            raise ValueError('Observation coordinates and ranges must be finite')
        if self.compute_volume and 48*self.volume_xy**2*self.volume_z > 1024**3:
            raise ValueError('Volume complex fields exceed the 1 GiB allocation limit; reduce volume samples')


@dataclass(frozen=True)
class SimulationConfig:
    objective: Objective = field(default_factory=Objective)
    illumination: Illumination = field(default_factory=Illumination)
    stack: Stack = field(default_factory=Stack)
    sampling: Sampling = field(default_factory=Sampling)

    def validate(self):
        self.objective.validate()
        self.illumination.validate()
        self.stack.validate()
        self.sampling.validate()
        if sum(self.stack.thicknesses[1:-1]) >= self.objective.focal_length:
            raise ValueError('The sum of finite layer thicknesses must be less than the focal length')
        if not np.isclose(self.objective.immersion_index, self.stack.indices[0]):
            raise ValueError('Objective immersion index must match the incident medium index')
