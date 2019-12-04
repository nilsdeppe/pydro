# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np
import ReconstructionPpao as recons

use_numba = True
try:
    import numba as nb
except:
    use_numba = False


def _ppao9531(primitive_vars, reconstruction_scheme, order_used):
    """
    Positivitiy-preserving adaptive-order reconstruction for the
    1d compressible Newtonian Euler equations.

    Uses 9th order unlimited reconstruction, supplemented by
    5th order unlimited, optimal 3rd order WCNS3Z, and first
    order reconstruction.
    """
    alpha9 = 4.0
    alpha5 = 4.0

    mass_density = primitive_vars[0]
    velocity = primitive_vars[1]
    pressure = primitive_vars[2]
    recons_mass_density = np.zeros(2 * len(mass_density) + 2)
    recons_velocity = np.zeros(2 * len(mass_density) + 2)
    recons_pressure = np.zeros(2 * len(mass_density) + 2)
    q = np.zeros(9)
    j = 4

    for i in range(j, len(mass_density) - j):
        q[0] = mass_density[i - 4]
        q[1] = mass_density[i - 3]
        q[2] = mass_density[i - 2]
        q[3] = mass_density[i - 1]
        q[4] = mass_density[i]
        q[5] = mass_density[i + 1]
        q[6] = mass_density[i + 2]
        q[7] = mass_density[i + 3]
        q[8] = mass_density[i + 4]
        # Reconstruct mass density, ensuring positivity

        successful_reconstruction = recons.adaptive_order_9(
            q, i, j, recons_mass_density, alpha9)
        if successful_reconstruction:
            order_used[i] = min(order_used[i], 9)
            successful_reconstruction = (
                recons_mass_density[2 * i + 1] > 0.0
                and recons_mass_density[2 * i + 2] > 0.0
                # Check j+3/2 and j-3/2
                and
                (3.021240234375e-3 * q[j - 4] - 4.0283203125e-2 * q[j - 3] +
                 4.229736328125e-1 * (q[j - 2] + q[j - 1]) -
                 3.5247802734375e-1 * q[j] + 1.69189453125e-1 * q[j + 1] -
                 6.04248046875e-2 * q[j + 2] + 1.3427734375e-2 * q[j + 3] -
                 1.373291015625e-3 * q[j + 4]) > 0.0 and
                (3.021240234375e-3 * q[j + 4] - 4.0283203125e-2 * q[j + 3] +
                 4.229736328125e-1 * (q[j + 2] + q[j + 1]) -
                 3.5247802734375e-1 * q[j] + 1.69189453125e-1 * q[j - 1] -
                 6.04248046875e-2 * q[j - 2] + 1.3427734375e-2 * q[j - 3] -
                 1.373291015625e-3 * q[j - 4]) > 0.0)
        if not successful_reconstruction:
            successful_reconstruction = recons.adaptive_order_5(
                q, i, j, recons_mass_density, keep_positive=True, alpha=alpha5)
            order_used[i] = min(order_used[i], 5)
            successful_reconstruction = (recons_mass_density[2 * i + 1] > 0.0
                                         and
                                         recons_mass_density[2 * i + 2] > 0.0)
        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 3)
            successful_reconstruction = recons.adaptive_order_wcns3z(
                q, i, j, recons_mass_density, keep_positive=True)
        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 1)
            recons.adaptive_order_1(q, i, j, recons_mass_density)

        # Reconstruct pressure
        q[0] = pressure[i - 4]
        q[1] = pressure[i - 3]
        q[2] = pressure[i - 2]
        q[3] = pressure[i - 1]
        q[4] = pressure[i]
        q[5] = pressure[i + 1]
        q[6] = pressure[i + 2]
        q[7] = pressure[i + 3]
        q[8] = pressure[i + 4]
        successful_reconstruction = recons.adaptive_order_9(
            q, i, j, recons_pressure, alpha9)
        if successful_reconstruction:
            order_used[i] = min(order_used[i], 9)
            successful_reconstruction = (
                recons_pressure[2 * i + 1] > 0.0
                and recons_pressure[2 * i + 2] > 0.0
                # Check j+3/2 and j-3/2
                and
                (3.021240234375e-3 * q[j - 4] - 4.0283203125e-2 * q[j - 3] +
                 4.229736328125e-1 * (q[j - 2] + q[j - 1]) -
                 3.5247802734375e-1 * q[j] + 1.69189453125e-1 * q[j + 1] -
                 6.04248046875e-2 * q[j + 2] + 1.3427734375e-2 * q[j + 3] -
                 1.373291015625e-3 * q[j + 4]) > 0.0 and
                (3.021240234375e-3 * q[j + 4] - 4.0283203125e-2 * q[j + 3] +
                 4.229736328125e-1 * (q[j + 2] + q[j + 1]) -
                 3.5247802734375e-1 * q[j] + 1.69189453125e-1 * q[j - 1] -
                 6.04248046875e-2 * q[j - 2] + 1.3427734375e-2 * q[j - 3] -
                 1.373291015625e-3 * q[j - 4]) > 0.0)

        if not successful_reconstruction:
            successful_reconstruction = recons.adaptive_order_5(
                q, i, j, recons_pressure, keep_positive=True, alpha=alpha5)
            order_used[i] = min(order_used[i], 5)
            successful_reconstruction = (recons_pressure[2 * i + 1] > 0.0
                                         and recons_pressure[2 * i + 2] > 0.0)

        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 3)
            successful_reconstruction = recons.adaptive_order_wcns3z(
                q, i, j, recons_pressure, keep_positive=True)
        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 1)
            recons.adaptive_order_1(q, i, j, recons_pressure)

        # Reconstruct velocity
        q[0] = velocity[i - 4]
        q[1] = velocity[i - 3]
        q[2] = velocity[i - 2]
        q[3] = velocity[i - 1]
        q[4] = velocity[i]
        q[5] = velocity[i + 1]
        q[6] = velocity[i + 2]
        q[7] = velocity[i + 3]
        q[8] = velocity[i + 4]
        successful_reconstruction = recons.adaptive_order_9(q,
                                                            i,
                                                            j,
                                                            recons_velocity,
                                                            alpha=alpha9,
                                                            eps=1.0e-4)
        order_used[i] = min(order_used[i], 9)
        if not successful_reconstruction:
            successful_reconstruction = recons.adaptive_order_5(
                q,
                i,
                j,
                recons_velocity,
                keep_positive=False,
                alpha=alpha5,
                eps=1.0e-4)
            order_used[i] = min(order_used[i], 5)
        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 3)
            successful_reconstruction = recons.adaptive_order_wcns3z(
                q, i, j, recons_velocity, keep_positive=False)

    return [recons_mass_density, recons_velocity, recons_pressure]


def _ppao31(primitive_vars, reconstruction_scheme, order_used):
    """
    Positivitiy-preserving adaptive-order reconstruction for the
    1d compressible Newtonian Euler equations.

    Uses optimal 3rd order WCNS3Z reconstruction, supplemented by
    first order reconstruction.
    """
    mass_density = primitive_vars[0]
    velocity = primitive_vars[1]
    pressure = primitive_vars[2]
    recons_mass_density = np.zeros(2 * len(mass_density) + 2)
    recons_velocity = np.zeros(2 * len(mass_density) + 2)
    recons_pressure = np.zeros(2 * len(mass_density) + 2)
    q = np.zeros(9)
    j = 4

    for i in range(j, len(mass_density) - j):
        q[0] = mass_density[i - 4]
        q[1] = mass_density[i - 3]
        q[2] = mass_density[i - 2]
        q[3] = mass_density[i - 1]
        q[4] = mass_density[i]
        q[5] = mass_density[i + 1]
        q[6] = mass_density[i + 2]
        q[7] = mass_density[i + 3]
        q[8] = mass_density[i + 4]

        # Reconstruct mass density, ensuring positivity
        order_used[i] = min(order_used[i], 3)
        successful_reconstruction = recons.adaptive_order_wcns3z(
            q, i, j, recons_mass_density, keep_positive=True)
        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 1)
            recons.adaptive_order_1(q, i, j, recons_mass_density)

        # Reconstruct pressure
        q[0] = pressure[i - 4]
        q[1] = pressure[i - 3]
        q[2] = pressure[i - 2]
        q[3] = pressure[i - 1]
        q[4] = pressure[i]
        q[5] = pressure[i + 1]
        q[6] = pressure[i + 2]
        q[7] = pressure[i + 3]
        q[8] = pressure[i + 4]
        order_used[i] = min(order_used[i], 3)
        successful_reconstruction = recons.adaptive_order_wcns3z(
            q, i, j, recons_pressure, keep_positive=True)
        if not successful_reconstruction:
            order_used[i] = min(order_used[i], 1)
            recons.adaptive_order_1(q, i, j, recons_pressure)

        # Reconstruct velocity
        q[0] = velocity[i - 4]
        q[1] = velocity[i - 3]
        q[2] = velocity[i - 2]
        q[3] = velocity[i - 1]
        q[4] = velocity[i]
        q[5] = velocity[i + 1]
        q[6] = velocity[i + 2]
        q[7] = velocity[i + 3]
        q[8] = velocity[i + 4]
        order_used[i] = min(order_used[i], 3)
        successful_reconstruction = recons.adaptive_order_wcns3z(
            q, i, j, recons_velocity, keep_positive=False)

    return [recons_mass_density, recons_velocity, recons_pressure]


if use_numba:
    ppao9531 = nb.njit(_ppao9531)
    ppao31 = nb.njit(_ppao31)
else:
    ppao9531 = _ppao9531
    ppao31 = _ppao31
