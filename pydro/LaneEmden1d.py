# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np

use_numba = True
try:
    import numba as nb
except:
    use_numba = False

import Reconstruction as recons
import Derivative
import NewtonianEuler as ne


def _apply_mask_recons_prim_impl(density, velocity, mask, atmosphere_density):
    # Apply atmosphere again but on reconstructed values
    for i in range(len(mask)):
        if mask[i]:
            density[2 * i] = atmosphere_density
            density[2 * i + 1] = atmosphere_density
            velocity[2 * i] = 0.0
            velocity[2 * i + 1] = 0.0


if use_numba:
    _apply_mask_recons_prim = nb.njit(_apply_mask_recons_prim_impl)
else:
    _apply_mask_recons_prim = _apply_mask_recons_prim_impl


class NewtonianEuler1dLaneEmden:
    _reconstruct_prims = None

    _x = None
    _dx = None
    _x_faces = None

    _boundary_conditions = None
    _buffer_size = 9 * 2

    _reconstructor = None
    _reconstruction_scheme = None
    _deriv_scheme = None

    _order_used = None

    def __init__(self, reconstruct_prims, x, boundary_conditions,
                 reconstructor, reconstruction_scheme, deriv_scheme):
        self._reconstruct_prims = reconstruct_prims

        self._x = np.copy(x)
        self._dx = x[1] - x[0]
        self._x_faces = np.arange(0, len(self._x) + 1, 1) * self._dx

        self._boundary_conditions = boundary_conditions

        self._reconstructor = reconstructor
        self._reconstruction_scheme = reconstruction_scheme
        self._deriv_scheme = deriv_scheme

        self._order_used = np.zeros(len(x), dtype=int)

    def _reset_order_used(self):
        length = len(self._x)
        # set to 100 because during reconstruction we use the min of the
        # current order used and the order used by past variables. It seems
        # unlikely we will use a 100th order scheme.
        if len(self._order_used) != length:
            self._order_used = np.zeros(length, dtype=int) + 100
        else:
            self._order_used[:] = 100

    def get_dx(self):
        """
        Return the inter-grid-point spacing
        """
        return self._dx

    def get_order_used(self):
        """
        Returns the lowest order used for reconstruction.
        """
        lower_bound = (
            0 if self._boundary_conditions[0] == ne.BoundaryCondition.Constant
            else self._buffer_size)
        upper_bound = (
            len(self._order_used)
            if self._boundary_conditions[1] == ne.BoundaryCondition.Constant
            else len(self._order_used) - self._buffer_size)

        return self._order_used[lower_bound:upper_bound]

    @staticmethod
    def compute_sound_speed(conserved_vars):
        return 2.0 * conserved_vars[0] / (1.0 + 2.0 * conserved_vars[0])

    def _flux_deriv(self, numerical_fluxes_at_faces, cell_fluxes):
        return Derivative.differentiate_flux(self._deriv_scheme, self._dx,
                                             numerical_fluxes_at_faces,
                                             cell_fluxes, self._order_used)

    def _flux(self, mass_density, velocity, pressure, total_energy):
        mass_density_flux = mass_density * velocity
        momentum_density_flux = mass_density * velocity**2 + mass_density**2
        energy_flux = velocity * (mass_density * total_energy + pressure)
        return [mass_density_flux, momentum_density_flux, energy_flux]

    def _sources(self, mass_density, velocity):
        star_mask = self._x < (np.pi - 1.0e-14)
        dPhi_dr = np.full(len(self._x), 0.0)
        dPhi_dr[star_mask] = (
            -2.0 * (self._x[star_mask] * np.cos(self._x[star_mask]) -
                    np.sin(self._x[star_mask])) / self._x[star_mask]**2)

        mass_density_source = -2.0 / self._x * (mass_density * velocity)
        momentum_density_source = (-2.0 / self._x *
                                   (mass_density * velocity**2) -
                                   mass_density * dPhi_dr)
        return [
            mass_density_source, momentum_density_source,
            np.zeros(mass_density_source.shape)
        ]

    def __call__(self, evolved_vars, time):
        self._reset_order_used()

        # Evolved vars are (rho, rho v^r, rho E), but we completely ignore
        # rho E since we use a polytropic equation of state.

        # Apply atmosphere once
        atmosphere_density = 1.0e-20

        atmosphere_mask = evolved_vars[0] < 1.1 * atmosphere_density
        evolved_vars[0][atmosphere_mask] = atmosphere_density
        evolved_vars[1][atmosphere_mask] = 0.0

        primitive_vars = np.asarray([
            evolved_vars[0], evolved_vars[1] / evolved_vars[0],
            evolved_vars[0]**2
        ])

        # Reset total energy using equation of state
        evolved_vars[2] = (primitive_vars[2] +
                           0.5 * primitive_vars[1]**2) / primitive_vars[0]

        primitive_vars[0, 0:self._buffer_size] = np.sin(
            self._x[0:self._buffer_size]) / self._x[0:self._buffer_size]
        primitive_vars[1, 0:self._buffer_size] = 0.0

        recons_prim = self._reconstructor(primitive_vars,
                                          self._reconstruction_scheme,
                                          self._order_used)

        # Apply BCs
        recons_prim[0][0] = 1.0
        recons_prim[0][1] = np.sin(self._dx) / self._dx
        for i in range(1, self._buffer_size):
            recons_prim[0][2 * i] = np.sin(i * self._dx) / (i * self._dx)
            recons_prim[0][2 * i + 1] = np.sin(
                (i + 1.0) * self._dx) / ((i + 1.0) * self._dx)

        recons_prim[1][0:2 * self._buffer_size] = 0.0

        _apply_mask_recons_prim(recons_prim[0], recons_prim[1],
                                atmosphere_mask, atmosphere_density)

        # Compute pressure from polytropic EOS: p=K rho^Gamma = rha^2
        recons_prim[2] = recons_prim[0]**2

        # Compute conserved variables
        recons_conserved = ne.compute_conserved(recons_prim)

        # Compute fluxes at reconstructed points
        recons_flux = self._flux(recons_prim[0], recons_prim[1],
                                 recons_prim[2], recons_conserved[2])

        # Compute numerical fluxes.
        recons_sound_speed = 2.0 * recons_prim[2] / (recons_prim[0] +
                                                     2.0 * recons_prim[2])
        numerical_flux = ne.compute_numerical_flux(
            recons_prim[0], recons_prim[0] * recons_prim[1],
            recons_conserved[2], recons_flux[0], recons_flux[1],
            recons_flux[2], recons_prim[1], recons_prim[2], recons_sound_speed)

        dt_evolved_vars = (-1.0 * self._flux_deriv(
            numerical_flux,
            np.asarray(
                self._flux(primitive_vars[0], primitive_vars[1],
                           primitive_vars[2], evolved_vars[2]))) +
                           self._sources(primitive_vars[0], primitive_vars[1]))

        for i in range(3):
            dt_evolved_vars[i, 0:self._buffer_size] = 0.0
            dt_evolved_vars[i, -self._buffer_size:] = 0.0

        dt_evolved_vars[2, :] = 0.0

        return dt_evolved_vars
