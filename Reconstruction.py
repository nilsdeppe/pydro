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
class Scheme(enum.Enum):
    Minmod = enum.auto()
    Wcns3 = enum.auto()
    Weno3 = enum.auto()
    Wcns5 = enum.auto()
    Wcns5z = enum.auto()


def _reconstruct_work_(u, extents, dim, ghost_zones, func, scheme):
    recons = np.zeros(2 * len(u) + 2)
    # Reconstruction in x
    # || -- || -- || -- || -- || -- || -- ||
    # bn    ny    yy    yy    yy    yn    nb
    for i in range(ghost_zones[0], extents[0] - ghost_zones[0]):
        func(recons, u, i, 0, 0, 0, scheme)
    return recons


def _compute_face_values_minmod_(recons, v, i, j, k, dim_to_recons, scheme):
    if dim_to_recons == 0:
        a = v[i] - v[i - 1]
        b = v[i + 1] - v[i]
        if a * b < 0.0:
            slope = 0.0
        if abs(a) < abs(b):
            slope = a
        else:
            slope = b
        recons[2 * i + 1] = v[i] - 0.5 * slope
        recons[2 * i + 2] = v[i] + 0.5 * slope


def _reconstruct_minmod(u, extents, dim, scheme):
    return np.asarray(
        _reconstruct_work(u, np.asarray(extents), dim, np.asarray([1]),
                          _compute_face_values_minmod, scheme))


def _compute_face_values_wcns3_(recons, v, i, j, k, dim_to_recons, scheme):
    """
    Applies a WCNS3 (http://dx.doi.org/10.1016/j.compfluid.2012.09.001) or
    WENO3 reconstruction.
    """
    if dim_to_recons == 0:
        eps = 1.0e-17
        if scheme == Scheme.Weno3:
            C = np.asarray([1.0 / 3.0, 2.0 / 3.0])
        else:
            C = np.asarray([0.25, 0.75])
        exponent = 2
        beta = np.asarray([(v[i] - v[i - 1])**2, (v[i + 1] - v[i])**2])
        alpha_denom = np.asarray([
            (eps * (1 + abs(v[i]) + abs(v[i - 1])) + beta[0])**exponent,
            (eps * (1 + abs(v[i + 1]) + abs(v[i])) + beta[1])**exponent
        ])

        alpha_l = C[::-1] / alpha_denom
        w_l = alpha_l / np.sum(alpha_l)
        recons[2 * i + 1] = w_l[0] * (0.5 * v[i] + 0.5 * v[i - 1]) + w_l[1] * (
            1.5 * v[i] - 0.5 * v[i + 1])

        alpha_r = C / alpha_denom
        w_r = alpha_r / np.sum(alpha_r)
        recons[2 * i + 2] = w_r[0] * (1.5 * v[i] - 0.5 * v[i - 1]) + w_r[1] * (
            0.5 * v[i] + 0.5 * v[i + 1])


def _reconstruct_wcns3(u, extents, dim, scheme):
    return np.asarray(
        _reconstruct_work(u, np.asarray(extents), dim, np.asarray([1]),
                          _compute_face_values_wcns3, scheme))


def _wcns5_impl_(recons, q, i, scheme):
    c = np.asarray([1.0 / 16.0, 10.0 / 16.0, 5.0 / 16.0])
    eps_machine = 2.0e-16
    exponent = 16
    if scheme == Scheme.Wcns5z:
        beta = np.asarray(
            [(4.0 / 3.0) * q[0] * q[0] - (19.0 / 3.0) * q[0] * q[1] +
             (25.0 / 3.0) * q[1] * q[1] + (11.0 / 3.0) * q[0] * q[2] -
             (31.0 / 3.0) * q[1] * q[2] + (10.0 / 3.0) * q[2] * q[2],
             (4.0 / 3.0) * q[1] * q[1] - (13.0 / 3.0) * q[1] * q[2] +
             (13.0 / 3.0) * q[2] * q[2] + (5.0 / 3.0) * q[1] * q[3] -
             (13.0 / 3.0) * q[2] * q[3] + (4.0 / 3.0) * q[3] * q[3],
             (10.0 / 3.0) * q[2] * q[2] - (31.0 / 3.0) * q[2] * q[3] +
             (25.0 / 3.0) * q[3] * q[3] + (11.0 / 3.0) * q[2] * q[4] -
             (19.0 / 3.0) * q[3] * q[4] + (4.0 / 3.0) * q[4] * q[4]])
    else:
        beta = np.asarray([
            0.25 * (q[0] - 4.0 * q[1] + 3.0 * q[2])**2 +
            (q[0] - 2.0 * q[1] + q[2])**2,
            0.25 * (q[1] - q[3])**2 + (q[1] - 2.0 * q[3])**2, 0.25 *
            (3.0 * q[2] - 4.0 * q[3] + q[4])**2 + (q[2] - 2.0 * q[3] + q[4])**2
        ])
    epsilon = np.asarray([
        eps_machine * (1.0 + abs(q[0]) + abs(q[1]) + abs(q[2])),
        eps_machine * (1.0 + abs(q[1]) + abs(q[2]) + abs(q[3])),
        eps_machine * (1.0 + abs(q[2]) + abs(q[3]) + abs(q[4]))
    ])

    # Reconstruct left state
    alpha_l = c[::-1] / (beta + epsilon)**exponent
    omega_l = alpha_l / np.sum(alpha_l)
    q_l = np.asarray([
        -0.125 * q[0] + 0.75 * q[1] + 0.375 * q[2],
        0.375 * q[1] + 0.75 * q[2] - 0.125 * q[3],
        1.875 * q[2] - 1.25 * q[3] + 0.375 * q[4]
    ])

    recons[2 * i + 1] = np.sum(omega_l * q_l)

    # Reconstruct right state
    alpha_r = c / (beta + epsilon)**exponent
    omega_r = alpha_r / np.sum(alpha_r)
    q_r = np.asarray([
        0.375 * q[0] - 1.25 * q[1] + 1.875 * q[2],
        -0.125 * q[1] + 0.75 * q[2] + 0.375 * q[3],
        0.375 * q[2] + 0.75 * q[3] - 0.125 * q[4]
    ])

    recons[2 * i + 2] = np.sum(omega_r * q_r)


def _compute_face_values_wcns5_(recons, q, i, j, k, dim_to_recons, scheme):
    """
    Applies a WCNS5 (http://dx.doi.org/10.1016/j.compfluid.2012.09.001,
    http://dx.doi.org/10.1016/j.compfluid.2015.08.023) or
    WCNS5Z reconstruction. In the WCNS5Z reconstruction the WENO5Z oscillation
    indicator is used.
    """
    if dim_to_recons == 0:
        _wcns5_impl(recons,
                    np.asarray([q[i - 2], q[i - 1], q[i], q[i + 1], q[i + 2]]),
                    i, scheme)


def _reconstruct_wcns5(u, extents, dim, scheme):
    return np.asarray(
        _reconstruct_work(u, np.asarray(extents), dim, np.asarray([2]),
                          _compute_face_values_wcns5, scheme))


if use_numba:
    _reconstruct_work = nb.jit(nopython=True)(_reconstruct_work_)
    _compute_face_values_minmod = nb.jit(
        nopython=True)(_compute_face_values_minmod_)
    _compute_face_values_wcns3 = nb.jit(
        nopython=True)(_compute_face_values_wcns3_)
    _wcns5_impl = nb.jit(nopython=True)(_wcns5_impl_)
    _compute_face_values_wcns5 = nb.jit(
        nopython=True)(_compute_face_values_wcns5_)
else:
    print("Please install Numba for better performance.")
    _reconstruct_work = _reconstruct_work_
    _compute_face_values_minmod = _compute_face_values_minmod_
    _compute_face_values_wcns3 = _compute_face_values_wcns3_
    _wcns5_impl = _wcns5_impl_
    _compute_face_values_wcns5 = _compute_face_values_wcns5_

_recons_dispatch = {
    Scheme.Minmod: _reconstruct_minmod,
    Scheme.Wcns3: _reconstruct_wcns3,
    Scheme.Weno3: _reconstruct_wcns3,
    Scheme.Wcns5: _reconstruct_wcns5,
    Scheme.Wcns5z: _reconstruct_wcns5
}


def reconstruct(vars_to_reconstruct, scheme):
    reconstructed_vars = [None] * len(vars_to_reconstruct)
    for i in range(len(vars_to_reconstruct)):
        extents = np.asarray([len(vars_to_reconstruct[i])])
        reconstructed_vars[i] = _recons_dispatch[scheme](
            vars_to_reconstruct[i], extents, 1, scheme)
    return reconstructed_vars
