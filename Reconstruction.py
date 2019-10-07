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
    Wcns5Weno = enum.auto()


def _reconstruct_work_(u, extents, dim, ghost_zones, func, scheme, order_used):
    recons = np.zeros(2 * len(u) + 2)
    # Reconstruction in x
    # || -- || -- || -- || -- || -- || -- ||
    # bn    ny    yy    yy    yy    yn    nb
    for i in range(ghost_zones, extents[0] - ghost_zones):
        order_used[i] = min(func(recons, u, i, 0, 0, 0, scheme), order_used[i])
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
        return 2


def _reconstruct_minmod(u, extents, dim, scheme, order_used):
    return np.asarray(
        _reconstruct_work(u, extents, dim, 1, _compute_face_values_minmod,
                          scheme, order_used))


def _compute_face_values_wcns3_(recons, v, i, j, k, dim_to_recons, scheme):
    """
    Applies a WCNS3 (http://dx.doi.org/10.1016/j.compfluid.2012.09.001) or
    WENO3 reconstruction.
    """
    if dim_to_recons == 0:
        eps = 1.0e-17
        if scheme == Scheme.Weno3:
            C0 = 1.0 / 3.0
            C1 = 2.0 / 3.0
        else:
            C0 = 0.25
            C1 = 0.75
        exponent = 2
        beta0 = (v[i] - v[i - 1])**2
        beta1 = (v[i + 1] - v[i])**2
        alpha_denom0 = (eps * (1 + abs(v[i]) + abs(v[i - 1])) +
                        beta0)**exponent
        alpha_denom1 = (eps * (1 + abs(v[i + 1]) + abs(v[i])) +
                        beta1)**exponent

        alpha_l0 = C1 / alpha_denom0
        alpha_l1 = C0 / alpha_denom1
        w_l0 = alpha_l0 / (alpha_l0 + alpha_l1)
        w_l1 = alpha_l1 / (alpha_l0 + alpha_l1)
        recons[2 * i + 1] = w_l0 * (0.5 * v[i] + 0.5 * v[i - 1]) + w_l1 * (
            1.5 * v[i] - 0.5 * v[i + 1])

        alpha_r0 = C0 / alpha_denom0
        alpha_r1 = C1 / alpha_denom1
        w_r0 = alpha_r0 / (alpha_r0 + alpha_r1)
        w_r1 = alpha_r1 / (alpha_r0 + alpha_r1)
        recons[2 * i + 2] = w_r0 * (1.5 * v[i] - 0.5 * v[i - 1]) + w_r1 * (
            0.5 * v[i] + 0.5 * v[i + 1])
        return 3


def _reconstruct_wcns3(u, extents, dim, scheme, order_used):
    return np.asarray(
        _reconstruct_work(u, extents, dim, 1, _compute_face_values_wcns3,
                          scheme, order_used))


def _wcns5_impl_(recons, q0, q1, q2, q3, q4, i, scheme):
    c0 = 1.0 / 16.0
    c1 = 10.0 / 16.0
    c2 = 5.0 / 16.0
    eps_machine = 2.0e-16
    exponent = 2
    epsilon0 = eps_machine * (1.0 + abs(q0) + abs(q1) + abs(q2))
    epsilon1 = eps_machine * (1.0 + abs(q1) + abs(q2) + abs(q3))
    epsilon2 = eps_machine * (1.0 + abs(q2) + abs(q3) + abs(q4))
    if scheme == Scheme.Wcns5z or scheme == Scheme.Wcns5Weno:
        beta0 = ((4.0 / 3.0) * q0 * q0 - (19.0 / 3.0) * q0 * q1 +
                 (25.0 / 3.0) * q1 * q1 + (11.0 / 3.0) * q0 * q2 -
                 (31.0 / 3.0) * q1 * q2 + (10.0 / 3.0) * q2 * q2)
        beta1 = ((4.0 / 3.0) * q1 * q1 - (13.0 / 3.0) * q1 * q2 +
                 (13.0 / 3.0) * q2 * q2 + (5.0 / 3.0) * q1 * q3 -
                 (13.0 / 3.0) * q2 * q3 + (4.0 / 3.0) * q3 * q3)
        beta2 = ((10.0 / 3.0) * q2 * q2 - (31.0 / 3.0) * q2 * q3 +
                 (25.0 / 3.0) * q3 * q3 + (11.0 / 3.0) * q2 * q4 -
                 (19.0 / 3.0) * q3 * q4 + (4.0 / 3.0) * q4 * q4)

    if scheme == Scheme.Wcns5z:
        tau5 = abs(beta2 - beta0)
        beta0 = (beta0 + epsilon0) / (beta0 + tau5 + epsilon0)
        beta1 = (beta1 + epsilon1) / (beta1 + tau5 + epsilon1)
        beta2 = (beta2 + epsilon2) / (beta2 + tau5 + epsilon2)

    elif scheme == Scheme.Wcns5:
        beta0 = 0.25 * (q0 - 4.0 * q1 + 3.0 * q2)**2 + (q0 - 2.0 * q1 + q2)**2
        beta1 = 0.25 * (q1 - q3)**2 + (q1 - 2.0 * q3)**2
        beta2 = 0.25 * (3.0 * q2 - 4.0 * q3 + q4)**2 + (q2 - 2.0 * q3 + q4)**2

    # Reconstruct left state
    alpha_l0 = c2 / (beta0 + epsilon0)**exponent
    alpha_l1 = c1 / (beta1 + epsilon1)**exponent
    alpha_l2 = c0 / (beta2 + epsilon2)**exponent
    omega_l0 = alpha_l0 / (alpha_l0 + alpha_l1 + alpha_l2)
    omega_l1 = alpha_l1 / (alpha_l0 + alpha_l1 + alpha_l2)
    omega_l2 = alpha_l2 / (alpha_l0 + alpha_l1 + alpha_l2)
    q_l0 = -0.125 * q0 + 0.75 * q1 + 0.375 * q2
    q_l1 = 0.375 * q1 + 0.75 * q2 - 0.125 * q3
    q_l2 = 1.875 * q2 - 1.25 * q3 + 0.375 * q4

    recons[2 * i + 1] = omega_l0 * q_l0 + omega_l1 * q_l1 + omega_l2 * q_l2

    # Reconstruct right state
    alpha_r0 = c0 / (beta0 + epsilon0)**exponent
    alpha_r1 = c1 / (beta1 + epsilon1)**exponent
    alpha_r2 = c2 / (beta2 + epsilon2)**exponent
    omega_r0 = alpha_r0 / (alpha_r0 + alpha_r1 + alpha_r2)
    omega_r1 = alpha_r1 / (alpha_r0 + alpha_r1 + alpha_r2)
    omega_r2 = alpha_r2 / (alpha_r0 + alpha_r1 + alpha_r2)
    q_r0 = 0.375 * q0 - 1.25 * q1 + 1.875 * q2
    q_r1 = -0.125 * q1 + 0.75 * q2 + 0.375 * q3
    q_r2 = 0.375 * q2 + 0.75 * q3 - 0.125 * q4

    recons[2 * i + 2] = omega_r0 * q_r0 + omega_r1 * q_r1 + omega_r2 * q_r2
    return 5


def _compute_face_values_wcns5_(recons, q, i, j, k, dim_to_recons, scheme):
    """
    Applies a WCNS5 (http://dx.doi.org/10.1016/j.compfluid.2012.09.001,
    http://dx.doi.org/10.1016/j.compfluid.2015.08.023) or
    WCNS5Z reconstruction. In the WCNS5Z reconstruction the WENO5Z oscillation
    indicator is used.
    """
    if dim_to_recons == 0:
        return _wcns5_impl(recons, q[i - 2], q[i - 1], q[i], q[i + 1],
                           q[i + 2], i, scheme)


def _reconstruct_wcns5(u, extents, dim, scheme, order_used):
    return np.asarray(
        _reconstruct_work(u, extents, dim, 2, _compute_face_values_wcns5,
                          scheme, order_used))



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
    Scheme.Wcns5z: _reconstruct_wcns5,
    Scheme.Wcns5Weno: _reconstruct_wcns5
}


def reconstruct(vars_to_reconstruct, scheme, order_used):
    reconstructed_vars = [None] * len(vars_to_reconstruct)
    for i in range(len(vars_to_reconstruct)):
        extents = np.asarray([len(vars_to_reconstruct[i])])
        reconstructed_vars[i] = _recons_dispatch[scheme](
            vars_to_reconstruct[i], np.asarray(extents), 1, scheme, order_used)
    return np.asarray(reconstructed_vars)
