# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import enum
import numpy as np
use_numba = True
try:
    import numba as nb
except:
    use_numba = False


@enum.unique
class Symmetry(enum.Enum):
    No = enum.auto()
    Cylindrical = enum.auto()
    Spherical = enum.auto()


@enum.unique
class NumericalFlux(enum.Enum):
    Rusanov = enum.auto()
    Hll = enum.auto()


_gamma = None
_numerical_flux = None
_symmetry_alpha = None


def set_gamma(gamma):
    global _gamma
    _gamma = gamma


def set_numerical_flux(numerical_flux):
    global _numerical_flux
    _numerical_flux = numerical_flux


def set_symmetry(symmetry):
    global _symmetry_alpha
    if symmetry == Symmetry.No:
        _symmetry_alpha = int(0)
    elif symmetry == Symmetry.Cylindrical:
        _symmetry_alpha = int(1)
    elif symmetry == Symmetry.Spherical:
        _symmetry_alpha = int(2)
    else:
        raise ValueError("Unknown symmetry option %s" % symmetry)


def get_symmetry():
    return _symmetry_alpha


def compute_pressure(mass_density, momentum_density, energy_density):
    return (_gamma - 1.0) * (energy_density -
                             0.5 * momentum_density**2 / mass_density)


def compute_energy_density(mass_density, momentum_density, pressure):
    return pressure / (_gamma - 1.0) + 0.5 * momentum_density**2 / mass_density


def compute_sound_speed(mass_density, press):
    return np.sqrt(_gamma * press / mass_density)


def compute_flux(evolved_vars):
    mass_density = evolved_vars[0]
    momentum_density = evolved_vars[1]
    energy_density = evolved_vars[2]
    pressure = compute_pressure(mass_density, momentum_density, energy_density)
    mass_density_flux = momentum_density
    momentum_density_flux = momentum_density**2 / mass_density + pressure
    energy_density_flux = (energy_density +
                           pressure) * momentum_density / mass_density
    return np.asarray(
        [mass_density_flux, momentum_density_flux, energy_density_flux])


def _rusanov_helper(lf_speed, mass_f, momentum_f, energy_f,
                    reconstructed_mass_density, reconstructed_momentum_density,
                    reconstructed_energy_density):
    nf_mass = np.zeros(len(reconstructed_mass_density) // 2)
    nf_momentum = np.zeros(len(reconstructed_mass_density) // 2)
    nf_energy = np.zeros(len(reconstructed_mass_density) // 2)

    for i in range(0, len(nf_mass), 1):
        speed = max(lf_speed[2 * i + 1], lf_speed[2 * i])
        nf_mass[i] = 0.5 * (mass_f[2 * i + 1] + mass_f[
            2 * i]) - 0.5 * speed * (reconstructed_mass_density[2 * i + 1] -
                                     reconstructed_mass_density[2 * i])
        nf_momentum[i] = 0.5 * (momentum_f[2 * i + 1] +
                                momentum_f[2 * i]) - 0.5 * speed * (
                                    reconstructed_momentum_density[2 * i + 1] -
                                    reconstructed_momentum_density[2 * i])
        nf_energy[i] = 0.5 * (energy_f[2 * i + 1] + energy_f[
            2 * i]) - 0.5 * speed * (reconstructed_energy_density[2 * i + 1] -
                                     reconstructed_energy_density[2 * i])

    return (nf_mass, nf_momentum, nf_energy)


def _hll_helper(v_p_cs, v_m_cs, mass_f, momentum_f, energy_f,
                reconstructed_mass_density, reconstructed_momentum_density,
                reconstructed_energy_density):
    nf_mass = np.zeros(len(reconstructed_mass_density) // 2)
    nf_momentum = np.zeros(len(reconstructed_mass_density) // 2)
    nf_energy = np.zeros(len(reconstructed_mass_density) // 2)

    for i in range(0, len(nf_mass), 1):
        # speed = max(lf_speed[2 * i + 1], lf_speed[2 * i])
        max_speed = max(max(v_p_cs[2 * i + 1], v_p_cs[2 * i]), 0.0)
        min_speed = min(min(v_m_cs[2 * i + 1], v_m_cs[2 * i]), 0.0)
        speed_diff = max_speed - min_speed
        nf_mass[i] = (-min_speed * mass_f[2 * i + 1] + max_speed *
                      mass_f[2 * i]) / speed_diff + max_speed * min_speed * (
                          reconstructed_mass_density[2 * i + 1] -
                          reconstructed_mass_density[2 * i]) / speed_diff
        nf_momentum[i] = (
            -min_speed * momentum_f[2 * i + 1] + max_speed *
            momentum_f[2 * i]) / speed_diff + max_speed * min_speed * (
                reconstructed_momentum_density[2 * i + 1] -
                reconstructed_momentum_density[2 * i]) / speed_diff
        nf_energy[i] = (-min_speed * energy_f[2 * i + 1] +
                        max_speed * energy_f[2 * i]
                        ) / speed_diff + max_speed * min_speed * (
                            reconstructed_energy_density[2 * i + 1] -
                            reconstructed_energy_density[2 * i]) / speed_diff

    return (nf_mass, nf_momentum, nf_energy)


# If numba is present, JIT the nuumerical flux helper
if use_numba:
    rusanov_helper = nb.jit(nopython=True)(_rusanov_helper)
    hll_helper = nb.jit(nopython=True)(_hll_helper)
else:
    rusanov_helper = _rusanov_helper
    hll_helper = _hll_helper


def compute_numerical_flux(mass_f, momentum_f, energy_f,
                           reconstructed_mass_density,
                           reconstructed_momentum_density,
                           reconstructed_energy_density):

    sound_speed = compute_sound_speed(
        reconstructed_mass_density,
        compute_pressure(reconstructed_mass_density,
                         reconstructed_momentum_density,
                         reconstructed_energy_density))

    velocity = reconstructed_momentum_density / reconstructed_mass_density
    if _numerical_flux == NumericalFlux.Rusanov:
        # Rusanov flux below
        lf_speed = abs(velocity) + sound_speed

        return rusanov_helper(lf_speed, mass_f, momentum_f, energy_f,
                              reconstructed_mass_density,
                              reconstructed_momentum_density,
                              reconstructed_energy_density)
    elif _numerical_flux == NumericalFlux.Hll:
        velocity_p_cs = velocity + sound_speed
        velocity_m_cs = velocity - sound_speed

        return hll_helper(velocity_p_cs, velocity_m_cs, mass_f, momentum_f,
                          energy_f, reconstructed_mass_density,
                          reconstructed_momentum_density,
                          reconstructed_energy_density)
    else:
        raise ValueError("Unknown numerical flux")


def compute_sources(radius, evolved_vars):
    """
    Computes source terms for the symmetry.
    """
    mass_density = evolved_vars[0]
    momentum_density = evolved_vars[1]
    energy_density = evolved_vars[2]
    factor = -_symmetry_alpha / radius
    pressure = compute_pressure(mass_density, momentum_density, energy_density)
    return (factor * momentum_density,
            factor * momentum_density**2 / mass_density, factor *
            (energy_density + pressure) * momentum_density / mass_density)


def compute_primitives(primitive_vars, evolved_vars):
    """
    primitive_vars[0] = rest mass density
    primitive_vars[1] = velocity in x-direction
    primitive_vars[2] = pressure

    evolved_vars[0] = rest mass density
    evolved_vars[1] = momentum density in x-direction
    evolved_vars[2] = energy density
    """
    if primitive_vars is None or not np.array_equal(primitive_vars.shape,
                                                    evolved_vars.shape):
        primitive_vars = np.zeros(evolved_vars.shape)

    primitive_vars[0] = evolved_vars[0]
    primitive_vars[1] = evolved_vars[1] / evolved_vars[0]
    primitive_vars[2] = (_gamma - 1.0) * (
        evolved_vars[2] - 0.5 * evolved_vars[1]**2 / evolved_vars[0])
    return primitive_vars


def compute_conserved(primitive_vars):
    """
    primitive_vars[0] = rest mass density
    primitive_vars[1] = velocity in x-direction
    primitive_vars[2] = pressure

    evolved_vars[0] = rest mass density
    evolved_vars[1] = momentum density in x-direction
    evolved_vars[2] = energy density
    """
    return np.asarray([
        primitive_vars[0], primitive_vars[1] * primitive_vars[0],
        compute_energy_density(primitive_vars[0],
                               primitive_vars[1] * primitive_vars[0],
                               primitive_vars[2])
    ])
