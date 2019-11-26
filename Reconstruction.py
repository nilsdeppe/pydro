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
    """
    An enum of the various different reconstruction
    schemes that are supported.
    """

    #: Minmod reconstruction.
    #:
    #: Minmod reconstruction is performed as
    #:
    #: .. math::
    #:   \begin{align}
    #:       \sigma_j=\mathrm{minmod} \left(\frac{q_i-q_{i-1}}{\Delta\xi},
    #:       \frac{q_{i+1}-q_i}{\Delta\xi}\right).
    #:   \end{align}
    #:
    #: where :math:`\Delta\xi` is the grid spacing and
    #: :math:`\mathrm{minmod}(a,b)` is defined as
    #:
    #: .. math::
    #:
    #:   \begin{align}
    #:     &\mathrm{minmod}(a,b)= \notag \\
    #:   &\left\{
    #:   \begin{array}{ll}
    #:     \mathrm{sgn}(a)\min(\lvert a\rvert, \lvert b\rvert)
    #:      & \mathrm{if} \; \mathrm{sgn}(a)=\mathrm{sgn}(b) \\
    #:     0 & \mathrm{otherwise}
    #:   \end{array}\right.
    #:  \end{align}
    #:
    #: The reconstructed solution at the faces is given by
    #:
    #: .. math::
    #:   \hat{q}_{i+1/2} = q_i +\frac{\Delta\xi}{2}\sigma_i
    #:
    #: See, e.g. section 9.3.1 of :cite:`RezzollaBook` for a discussion.
    Minmod = enum.auto()
    #: Third order weighted compact nonlinear scheme reconstruction
    #: :cite:`DENG200022`.
    #:
    #: Third order WCNS3 reconstruction is done by first defining
    #: oscillation indicators :math:`\beta_0` and :math:`\beta_1` as
    #:
    #: .. math::
    #:   \begin{align}
    #:     \beta_0 &= (q_i - q_{i-1})^2 \\
    #:     \beta_1 &= (q_{i+1} - q_{i})^2
    #:   \end{align}
    #:
    #: Then coefficients :math:`\alpha_k` are defined as
    #:
    #: .. math::
    #:   \alpha_k = \frac{c_k}{(\beta_k + \epsilon_k)^2}
    #:
    #: where :math:`\epsilon_k` is a factor used to avoid division
    #: by zero and is set to
    #:
    #: .. math::
    #:   \begin{align}
    #:     \epsilon_0 &= 10^{-17}\left(1 + |q_{i}| + |q_{i-1}|\right) \\
    #:     \epsilon_1 &= 10^{-17}\left(1 + |q_{i}| + |q_{i+1}|\right)
    #:   \end{align}
    #:
    #: and the linear weights are :math:`c_0=1/4` and :math:`c_1=3/4`.
    #: Finally, we define the nonlinear weights:
    #:
    #: .. math::
    #:   \omega_k=\frac{\alpha_k}{\sum_{k=0}^{1}\alpha_k}
    #:
    #: The reconstruction stencils are given by:
    #:
    #: .. math::
    #:   \begin{align}
    #:     q^0_{i+1/2}&=\frac{3}{2}q_i-\frac{1}{2}q_{i-1} \\
    #:     q^1_{i+1/2}&=\frac{1}{2}q_i+\frac{1}{2}q_{i+1}
    #:   \end{align}
    #:
    #: The final reconstructed solution is given by
    #:
    #: .. math::
    #:   \hat{q}_{i+1/2}=\sum_{k=0}^{1}\omega_k q^k_{i+1/2}
    Wcns3 = enum.auto()
    #: Third order weighted essentially non-oscillarity reconstruction.
    #:
    #: The same as the :py:meth:`Wcns3` reconstruction except with
    #: :math:`c_0=1/3` and :math:`c_1=2/3`.
    Weno3 = enum.auto()
    #: Fifth order weighted compact nonlinear scheme reconstruction
    #: :cite:`Nonomura20138`.
    #:
    #: The oscillation indicators are given by
    #:
    #: .. math::
    #:   \begin{align}
    #:     \beta_0 &= \frac{1}{4}\left(q_{i-2}-4 q_{i-1}+3 q_{i}\right)^2
    #:               + \left(q_{i-2}-2 q_{i-1} + q_{i}\right)^2 \\
    #:     \beta_1 &= \frac{1}{4}\left(q_{i-1} - q_{i+1}\right)^2
    #:               + \left(q_{i-1} - 2 q_{i+1}\right)^2 \\
    #:     \beta_2 &= \frac{1}{4}\left(3 q_{i}-4 q_{i+1}+q_{i+2}\right)^2
    #:               + \left(q_{i} - 2 q_{i+1} + q_{i+2}\right)^2
    #:   \end{align}
    #:
    #: Then coefficients :math:`\alpha_k` are defined as
    #:
    #: .. math::
    #:   \alpha_k = \frac{c_k}{(\beta_k + \epsilon_k)^2}
    #:
    #: where :math:`\epsilon_k` is a factor used to avoid division
    #: by zero and is set to
    #:
    #: .. math::
    #:   \begin{align}
    #:     \epsilon_0 &= 2\times10^{-16}\left(1 + |q_{i}| + |q_{i-1}|
    #:                   + |q_{i-2}|\right) \\
    #:     \epsilon_1 &= 2\times10^{-16}\left(1 + |q_{i}| + |q_{i+1}|
    #:                   + |q_{i-1}|\right) \\
    #:     \epsilon_2 &= 2\times10^{-16}\left(1 + |q_{i}| + |q_{i+1}|
    #:                   + |q_{i+2}|\right)
    #:   \end{align}
    #:
    #: and the linear weights are :math:`c_0=1/16, c_1=10/16`, and
    #: :math:`c_2=5/16`. Finally, we define the nonlinear weights:
    #:
    #: .. math::
    #:   \omega_k=\frac{\alpha_k}{\sum_{k=0}^{2}\alpha_k}
    #:
    #: The reconstruction stencils are given by:
    #:
    #: .. math::
    #:   \begin{align}
    #:     q^0_{i+1/2}&=\frac{3}{8}q_{i-2} - \frac{5}{4}q_{i-1}
    #:                  + \frac{15}{8}q_i, \\
    #:     q^1_{i+1/2}&=-\frac{1}{8}q_{i-1} + \frac{3}{4}q_i
    #:                  + \frac{3}{8}q_{i+1}, \\
    #:     q^2_{i+1/2}&=\frac{3}{8}q_i + \frac{3}{4}q_{i+1}
    #:                  - \frac{1}{8}q_{i+2}.
    #:   \end{align}
    #:
    #: The final reconstructed solution is given by
    #:
    #: .. math::
    #:   \hat{q}_{i+1/2}=\sum_{k=0}^{2}\omega_k q^k_{i+1/2}
    Wcns5 = enum.auto()
    #: Fifth order weighted compact nonlinear scheme reconstruction
    #: with the :math:`Z` oscillation indicator.
    #:
    #: Follows the procedure of :py:meth:`Wcns5` except using the
    #: oscillation indicators given by
    #:
    #: .. math::
    #:     \beta_k^Z =\frac{\beta_k+\epsilon_k}{\beta_k +
    #:                 \tau_5 + \epsilon_k}
    #:
    #: where
    #:
    #: .. math::
    #:   \tau_5 = |\beta_2 - \beta_0|
    #:
    #: and the oscillation indicators are the ones from Jiang and
    #: Shu :cite:`JIANG1996202`, as described in :py:meth:`Wcns5Weno`.
    Wcns5z = enum.auto()
    #: Fifth order weighted compact nonlinear scheme reconstruction
    #: with the Jiang and Shu :cite:`JIANG1996202` weights.
    #:
    #: Follows the procedure of :py:meth:`Wcns5` except using the
    #: oscillation indicators given by
    #:
    #: .. math::
    #:   \begin{align}
    #:     \beta_0 &=\frac{1}{4}\left(q_{i-2}-4q_{i-1}+3q_i\right)^2
    #:              +\frac{13}{12}\left(q_{i-2}-2q_{i-1}+q_i\right)^2 \\
    #:     \beta_1 &=\frac{1}{4}\left(q_{i-1}-q_{i+1}\right)^2
    #:              +\frac{13}{12}\left(q_{i-1} - 2 q_{i+1}\right)^2 \\
    #:     \beta_2 &=\frac{1}{4}\left(-3q_i+4q_{i+1}-q_{i+2}\right)^2
    #:              +\frac{13}{12}\left(q_i-2q_{i+1}+q_{i+2}\right)^2.
    #:   \end{align}
    #:
    Wcns5Weno = enum.auto()
    Mp5 = enum.auto()


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


def _reconstruct_mp5(u, extents, dim, scheme, order_used):
    return np.asarray(
        _reconstruct_work(u, extents, dim, 2, _compute_face_values_mp5, scheme,
                          order_used))


def _mp5_impl_(recons, q, i, scheme):
    def minmod(a, b):
        if a * b < 0.0:
            return 0.0
        if abs(a) < abs(b):
            return a
        else:
            return b

    def median(a, b, c):
        return a + minmod(b - a, c - a)

    alpha = 4.0

    j = 2
    d_jm1 = q[j - 2] + q[j] - 2.0 * q[j - 1]
    d_j = q[j - 1] + q[j + 1] - 2.0 * q[j]
    d_jp1 = q[j] + q[j + 2] - 2.0 * q[j + 1]
    d_mmm = minmod(d_j, d_jm1)
    d_mmp = minmod(d_j, d_jp1)

    # Reconstruct left state
    q_av = 0.5 * (q[j - 1] + q[j])
    q_fl = q_av - 0.5 * d_j
    q_md = q_av - 0.5 * d_mmm
    q_ul = q[j] + alpha * (q[j] + q[j + 1])
    q_lc = q_fl + 4. / 3. * d_mmp
    q_min = max(min(min(q[j], q[j + 1]), q_md), min(min(q[j], q_ul), q_lc))
    q_max = min(max(max(q[j], q[j + 1]), q_md), max(max(q[j], q_ul), q_lc))
    q_l = (-3.0 * q[j - 2] + 27.0 * q[j - 1] + 47.0 * q[j] - 13.0 * q[j + 1] +
           2.0 * q[j - 2]) / 60.0
    recons[2 * i + 1] = median(q_l, q_min, q_max)

    # Reconstruct right state
    q_av = 0.5 * (q[j] + q[j + 1])
    q_fl = q_av - 0.5 * d_j
    # q_fr = q_av - 0.5 * d_ip1
    q_md = q_av - 0.5 * d_mmp
    q_ul = q[j] + alpha * (q[j] - q[j - 1])
    q_lc = q_fl + 4. / 3. * d_mmm
    q_min = max(min(min(q[j], q[j + 1]), q_md), min(min(q[j], q_ul), q_lc))
    q_max = min(max(max(q[j], q[j + 1]), q_md), max(max(q[j], q_ul), q_lc))
    q_r = (2.0 * q[j - 2] - 13.0 * q[j - 1] + 47.0 * q[j] + 27.0 * q[j + 1] -
           3.0 * q[j + 2]) / 60.0
    recons[2 * i + 2] = median(q_r, q_min, q_max)
    return 5


def _compute_face_values_mp5_(recons, q, i, j, k, dim_to_recons, scheme):
    if dim_to_recons == 0:
        _mp5_impl(recons,
                  np.asarray([q[i - 2], q[i - 1], q[i], q[i + 1], q[i + 2]]),
                  i, scheme)


if use_numba:
    _reconstruct_work = nb.jit(nopython=True)(_reconstruct_work_)
    _compute_face_values_minmod = nb.jit(
        nopython=True)(_compute_face_values_minmod_)
    _compute_face_values_wcns3 = nb.jit(
        nopython=True)(_compute_face_values_wcns3_)
    _wcns5_impl = nb.jit(nopython=True)(_wcns5_impl_)
    _compute_face_values_wcns5 = nb.jit(
        nopython=True)(_compute_face_values_wcns5_)
    _mp5_impl = nb.jit(nopython=True)(_mp5_impl_)
    _compute_face_values_mp5 = nb.jit(nopython=True)(_compute_face_values_mp5_)
else:
    print("Please install Numba for better performance.")
    _reconstruct_work = _reconstruct_work_
    _compute_face_values_minmod = _compute_face_values_minmod_
    _compute_face_values_wcns3 = _compute_face_values_wcns3_
    _wcns5_impl = _wcns5_impl_
    _compute_face_values_wcns5 = _compute_face_values_wcns5_
    _mp5_impl = _mp5_impl_
    _compute_face_values_mp5 = _compute_face_values_mp5_

_recons_dispatch = {
    Scheme.Minmod: _reconstruct_minmod,
    Scheme.Wcns3: _reconstruct_wcns3,
    Scheme.Weno3: _reconstruct_wcns3,
    Scheme.Wcns5: _reconstruct_wcns5,
    Scheme.Wcns5z: _reconstruct_wcns5,
    Scheme.Wcns5Weno: _reconstruct_wcns5,
    Scheme.Mp5: _reconstruct_mp5,
}


def reconstruct(vars_to_reconstruct, scheme, order_used):
    """
    Reconstructs all variables using the requested scheme.

    :param vars_to_reconstruct: The variables at the cell centers.
    :type vars_to_reconstruct: list of list of double

    :param Scheme scheme: The reconstruction scheme to use.

    :param order_used: Filled by the function and is used to return
        the order of the reconstruction used.
    :type order_used: list of int

    :return: (`list of list of double`) The face reconstructed variables.
        Each variable is of length `2 * number_of_cells`
    """
    reconstructed_vars = [None] * len(vars_to_reconstruct)
    for i in range(len(vars_to_reconstruct)):
        extents = np.asarray([len(vars_to_reconstruct[i])])
        reconstructed_vars[i] = _recons_dispatch[scheme](
            vars_to_reconstruct[i], np.asarray(extents), 1, scheme, order_used)
    return np.asarray(reconstructed_vars)
