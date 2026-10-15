"""Objective pupil fields and optional Cartesian data import."""
from math import factorial
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .model import Objective, Illumination


def generate_pupil_coordinates(objective: Objective, n_phi: int, n_theta: int):
    """Return polar angles and Cartesian pupil coordinates in micrometres."""
    theta = np.linspace(0, objective.theta_max, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)  # Both endpoints receive half weight in integration.
    rho = objective.focal_length * np.sin(theta)
    p, r = np.meshgrid(phi, rho, indexing='ij')
    return theta, phi, r*np.cos(p), r*np.sin(p)


def zernike_phase(rho, phi, coefficients):
    """Unnormalised real Zernike polynomials, n=0..4, m=-n:2:n."""
    result = np.zeros(np.broadcast_shapes(np.shape(rho), np.shape(phi)))
    k = 0
    for n in range(5):
        for m in range(-n, n+1, 2):
            a = coefficients[k]
            k += 1
            if a == 0:
                continue
            am = abs(m)
            radial = sum(((-1)**s * factorial(n-s) /
                          (factorial(s)*factorial((n+am)//2-s)*factorial((n-am)//2-s))) * rho**(n-2*s)
                         for s in range((n-am)//2+1))
            angular = np.cos(m*phi) if m > 0 else np.sin(am*phi) if m < 0 else 1
            result = result + a*radial*angular
    return result


def generate_inc_field(objective: Objective, light: Illumination, n_phi: int, n_theta: int):
    """Return complex Cartesian pupil fields and their coordinate grids."""
    theta, phi, x, y = generate_pupil_coordinates(objective, n_phi, n_theta)
    p = phi[:, None]
    rho = objective.focal_length*np.sin(theta)[None, :]
    norm_rho = rho / (objective.focal_length*np.sin(objective.theta_max))
    envelope = np.exp(-(rho/light.waist)**2)
    phase = light.vortex_charge*p + zernike_phase(norm_rho, p, light.zernike)
    scalar = envelope*np.exp(1j*phase)
    if light.polarization == 'uniform':
        px = np.cos(np.deg2rad(light.gamma_deg))
        py = np.sin(np.deg2rad(light.gamma_deg))*np.exp(1j*np.deg2rad(light.beta_deg))
    elif light.polarization == 'spiral':
        px = np.cos(p + np.deg2rad(light.delta_deg))
        py = np.sin(p + np.deg2rad(light.delta_deg))
    else:
        raise ValueError('Unknown polarization')
    return np.broadcast_to(scalar*px, x.shape).copy(), np.broadcast_to(scalar*py, x.shape).copy(), x, y


def load_inc_field(ex_file, ey_file, x_file, y_file, objective, n_phi, n_theta):
    """Text complex fields in (y,x) order; monotonic x/y vectors in micrometres."""
    x = np.loadtxt(x_file).ravel()
    y = np.loadtxt(y_file).ravel()
    if len(x) < 2 or len(y) < 2 or not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError('Cartesian coordinate vectors must be strictly increasing')
    _, _, xp, yp = generate_pupil_coordinates(objective, n_phi, n_theta)
    sample = np.stack((yp.ravel(), xp.ravel()), axis=-1)
    fields = []
    for path in (ex_file, ey_file):
        field = np.loadtxt(path, dtype=complex, converters=lambda token: complex(token.decode().strip('()').replace('i', 'j') if isinstance(token, bytes) else token.strip('()').replace('i', 'j')))
        if field.shape != (len(y), len(x)):
            raise ValueError(f'{path}: expected field shape (len(y), len(x))')
        interp = RegularGridInterpolator((y, x), field, bounds_error=False, fill_value=0)
        fields.append(interp(sample).reshape(xp.shape))
    return *fields, xp, yp
