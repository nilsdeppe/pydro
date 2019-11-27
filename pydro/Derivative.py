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
    An enum of the various different differentiation routines
    that are supported.
    """

    #: Second-order finite-difference derivative :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \frac{\partial q_i}{\partial x}\approx
    #:   \frac{q_{i+1/2}-q_{i-1/2}}{\Delta x}
    MD = enum.auto()
    #: Fourth-order finite-difference derivative using only face values
    #: :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \frac{\partial q_i}{\partial x}\approx
    #:   \frac{1}{\Delta x}\left[\frac{9}{8}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{1}{24}(q_{i+3/2}-q_{i-3/2})\right]
    MD4 = enum.auto()
    #: Sixth-order finite-difference derivative using only face values
    #: :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \begin{align}
    #:   \frac{\partial q_i}{\partial x}&\approx
    #:   \frac{1}{\Delta x}\left[
    #:   \frac{75}{64}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{25}{384}(q_{i+3/2}-q_{i-3/2})\right. \\
    #:   &\left.+\frac{3}{640}(q_{i+5/2}-q_{i-5/2})\right]
    #:   \end{align}
    MD6 = enum.auto()
    #: Eighth-order finite-difference derivative using only face values
    #: :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \begin{align}
    #:   \frac{\partial q_i}{\partial x}&\approx
    #:   \frac{1}{\Delta x}\left[
    #:   \frac{1225}{1024}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{245}{3072}(q_{i+3/2}-q_{i-3/2})\right. \\
    #:   &\left.+\frac{49}{5120}(q_{i+5/2}-q_{i-5/2})-
    #:   \frac{5}{7168}(q_{i+7/2}-q_{i-7/2})\right]
    #:   \end{align}
    MD8 = enum.auto()
    #: Tenth-order finite-difference derivative using only face values
    #: :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \begin{align}
    #:   \frac{\partial q_i}{\partial x}&\approx
    #:   \frac{1}{\Delta x}\left[
    #:   \frac{19845}{16384}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{735}{8192}(q_{i+3/2}-q_{i-3/2})\right. \\
    #:   &\left.+\frac{567}{40960}(q_{i+5/2}-q_{i-5/2})-
    #:   \frac{405}{229376}(q_{i+7/2}-q_{i-7/2})\right. \\
    #:   &\left.+\frac{35}{294912}(q_{i+9/2}-q_{i-9/2})\right]
    #:   \end{align}
    MD10 = enum.auto()
    #: Variable-order finite-difference derivative using only face
    #: values.
    #:
    #: The order is adjusted according to the `order_used` argument
    #: passed to :py:func:`Derivative.differentiate_flux`
    MDV = enum.auto()
    #: Fourth-order finite-difference derivative using face and node
    #: values (MND) :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \frac{\partial q_i}{\partial x}\approx
    #:   \frac{1}{\Delta x}\left[\frac{4}{3}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{1}{6}(q_{i+1}-q_{i-1})\right]
    MND4 = enum.auto()
    #: Sixth-order finite-difference derivative using face and node
    #: values (MND) :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \frac{\partial q_i}{\partial x}&\approx
    #:   \frac{1}{\Delta x}\left[\frac{3}{2}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{3}{10}(q_{i+1}-q_{i-1})\right. \\
    #:   &\left.+\frac{1}{30}(q_{i+3/2}-q_{i-3/2})\right]
    MND6 = enum.auto()
    #: Eigth-order finite-difference derivative using face and node
    #: values (MND) :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \frac{\partial q_i}{\partial x}&\approx
    #:   \frac{1}{\Delta x}\left[\frac{8}{5}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{2}{5}(q_{i+1}-q_{i-1})\right. \\
    #:   &\left.+\frac{8}{105}(q_{i+3/2}-q_{i-3/2})-
    #:   \frac{1}{140}(q_{i+2}-q_{i-2})\right]
    MND8 = enum.auto()
    #: Tenth-order finite-difference derivative using face and node
    #: values (MND) :cite:`Nonomura20138`
    #:
    #: .. math::
    #:   \frac{\partial q_i}{\partial x}&\approx
    #:   \frac{1}{\Delta x}\left[\frac{5}{3}(q_{i+1/2}-q_{i-1/2})-
    #:   \frac{10}{21}(q_{i+1}-q_{i-1})\right. \\
    #:   &\left.+\frac{5}{42}(q_{i+3/2}-q_{i-3/2})-
    #:   \frac{5}{252}(q_{i+2}-q_{i-2})\right. \\
    #:   &\left.+\frac{1}{630}(q_{i+5/2}-q_{i-5/2})\right]
    MND10 = enum.auto()
    #: Variable-order finite-difference derivative using face and node
    #: values (MND).
    #:
    #: The order is adjusted according to the `order_used` argument
    #: passed to :py:func:`Derivative.differentiate_flux`
    MNDV = enum.auto()


def _md_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 2nd order finite difference using numerical fluxes
    """
    result = []
    inv_dx = 1.0 / dx
    for i in range(len(numerical_flux)):
        result.append(inv_dx *
                      (numerical_flux[i][1:] - numerical_flux[i][:-1]))
    return result


def _md4_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 4th order finite difference using numerical fluxes
    """
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(numerical_flux[j]) - 2):
            temp[i] = 1.125 * (numerical_flux[j][i + 1] - numerical_flux[j][i]
                               ) * inv_dx - 0.041666666666666664 * (
                                   numerical_flux[j][i + 2] -
                                   numerical_flux[j][i - 1]) * inv_dx
        result.append(temp)
    return result


def _md6_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 6th order finite difference using numerical fluxes
    """
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(2, len(numerical_flux[j]) - 3):
            temp[i] = (
                1.171875 * (numerical_flux[j][i + 1] - numerical_flux[j][i]) -
                0.06510416666666667 *
                (numerical_flux[j][i + 2] - numerical_flux[j][i - 1]) +
                0.0046875 *
                (numerical_flux[j][i + 3] - numerical_flux[j][i - 2])) * inv_dx
        result.append(temp)
    return result


def _md8_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 8th order finite difference using numerical fluxes
    """
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(3, len(numerical_flux[j]) - 4):
            temp[i] = (
                1.1962890625 *
                (numerical_flux[j][i + 1] - numerical_flux[j][i]) -
                0.07975260416666667 *
                (numerical_flux[j][i + 2] - numerical_flux[j][i - 1]) +
                0.0095703125 *
                (numerical_flux[j][i + 3] - numerical_flux[j][i - 2]) -
                0.0006975446428571429 *
                (numerical_flux[j][i + 4] - numerical_flux[j][i - 3])) * inv_dx
        result.append(temp)
    return result


def _md10_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 10th order finite difference using numerical fluxes
    """
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(4, len(numerical_flux[j]) - 5):
            temp[i] = (
                1.21124267578125 *
                (numerical_flux[j][i + 1] - numerical_flux[j][i]) -
                0.0897216796875 *
                (numerical_flux[j][i + 2] - numerical_flux[j][i - 1]) +
                0.0138427734375 *
                (numerical_flux[j][i + 3] - numerical_flux[j][i - 2]) -
                0.0017656598772321428 *
                (numerical_flux[j][i + 4] - numerical_flux[j][i - 3]) +
                0.00011867947048611111 *
                (numerical_flux[j][i + 5] - numerical_flux[j][i - 4])) * inv_dx
        result.append(temp)
    return result


def _mdv_impl_(dx, numerical_flux, center_flux, order_used):
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(4, len(center_flux[j]) - 5):
            if order_used[i] > 8:
                # Tenth order
                temp[i] = (
                    1.21124267578125 *
                    (numerical_flux[j][i + 1] - numerical_flux[j][i]) -
                    0.0897216796875 *
                    (numerical_flux[j][i + 2] - numerical_flux[j][i - 1]) +
                    0.0138427734375 *
                    (numerical_flux[j][i + 3] - numerical_flux[j][i - 2]) -
                    0.0017656598772321428 *
                    (numerical_flux[j][i + 4] - numerical_flux[j][i - 3]) +
                    0.00011867947048611111 *
                    (numerical_flux[j][i + 5] - numerical_flux[j][i - 4])
                ) * inv_dx
            elif order_used[i] > 6:
                # Eighth order
                temp[i] = (
                    1.1962890625 *
                    (numerical_flux[j][i + 1] - numerical_flux[j][i]) -
                    0.07975260416666667 *
                    (numerical_flux[j][i + 2] - numerical_flux[j][i - 1]) +
                    0.0095703125 *
                    (numerical_flux[j][i + 3] - numerical_flux[j][i - 2]) -
                    0.0006975446428571429 *
                    (numerical_flux[j][i + 4] - numerical_flux[j][i - 3])
                ) * inv_dx
            elif order_used[i] > 4:
                # Sixth order
                temp[i] = (
                    1.171875 *
                    (numerical_flux[j][i + 1] - numerical_flux[j][i]) -
                    0.06510416666666667 *
                    (numerical_flux[j][i + 2] - numerical_flux[j][i - 1]) +
                    0.0046875 *
                    (numerical_flux[j][i + 3] - numerical_flux[j][i - 2])
                ) * inv_dx
            elif order_used[i] > 2:
                # Fourth order
                temp[i] = 1.125 * (numerical_flux[j][i + 1] - numerical_flux[j]
                                   [i]) * inv_dx - 0.041666666666666664 * (
                                       numerical_flux[j][i + 2] -
                                       numerical_flux[j][i - 1]) * inv_dx
            else:
                f_p = numerical_flux[j][i + 1]
                f_m = numerical_flux[j][i]
                temp[i] = (f_p - f_m) * inv_dx
        result.append(temp)
    return result


def _mnd4_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 4th order midpoint-and-node-to-node differencing of
    http://dx.doi.org/10.1016/j.compfluid.2012.09.001
    """
    if center_flux is None:
        raise TypeError("Center flux must not be None")
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(center_flux[j]) - 1):
            f_p = 1.3333333333333333 * numerical_flux[j][
                i + 1] - 0.16666666666666666 * center_flux[j][i + 1]
            f_m = 1.3333333333333333 * numerical_flux[j][
                i] - 0.16666666666666666 * center_flux[j][i - 1]
            temp[i] = (f_p - f_m) * inv_dx
        result.append(temp)
    return result


def _mnd6_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 6th order midpoint-and-node-to-node differencing of
    http://dx.doi.org/10.1016/j.compfluid.2012.09.001
    """
    if center_flux is None:
        raise TypeError("Center flux must not be None")
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(center_flux[j]) - 2):
            f_p = 1.5 * numerical_flux[j][i + 1] - 0.3 * center_flux[j][
                i + 1] + 0.03333333333333333 * numerical_flux[j][i + 2]
            f_m = 1.5 * numerical_flux[j][i] - 0.3 * center_flux[j][
                i - 1] + 0.03333333333333333 * numerical_flux[j][i - 1]
            temp[i] = (f_p - f_m) * inv_dx
        result.append(temp)
    return result


def _mnd8_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 8th order midpoint-and-node-to-node differencing of
    http://dx.doi.org/10.1016/j.compfluid.2012.09.001
    """
    if center_flux is None:
        raise TypeError("Center flux must not be None")
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(center_flux[j]) - 2):
            f_p = 1.6 * numerical_flux[j][i + 1] - 0.4 * center_flux[j][
                i + 1] + 0.0761904761904762 * numerical_flux[j][
                    i + 2] - 0.007142857142857143 * center_flux[j][i + 2]
            f_m = 1.6 * numerical_flux[j][i] - 0.4 * center_flux[j][
                i - 1] + 0.0761904761904762 * numerical_flux[j][
                    i - 1] - 0.007142857142857143 * center_flux[j][i - 2]
            temp[i] = (f_p - f_m) * inv_dx
        result.append(temp)
    return result


def _mnd10_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 10th order midpoint-and-node-to-node differencing of
    http://dx.doi.org/10.1016/j.compfluid.2012.09.001
    """
    if center_flux is None:
        raise TypeError("Center flux must not be None")
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(center_flux[j]) - 2):
            f_p = 1.6666666666666667 * numerical_flux[j][
                i + 1] - 0.47619047619047616 * center_flux[j][
                    i + 1] + 0.11904761904761904 * numerical_flux[j][
                        i + 2] - 0.01984126984126984 * center_flux[j][
                            i + 2] + 0.0015873015873015873 * numerical_flux[j][
                                i + 3]
            f_m = 1.6666666666666667 * numerical_flux[j][
                i] - 0.47619047619047616 * center_flux[j][
                    i - 1] + 0.11904761904761904 * numerical_flux[j][
                        i - 1] - 0.01984126984126984 * center_flux[j][
                            i - 2] + 0.0015873015873015873 * numerical_flux[j][
                                i - 2]
            temp[i] = (f_p - f_m) * inv_dx
        result.append(temp)
    return result


def _mndv_impl_(dx, numerical_flux, center_flux, order_used):
    if center_flux is None:
        raise TypeError("Center flux must not be None")
    result = []
    inv_dx = 1.0 / dx
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(4, len(center_flux[j]) - 5):
            if order_used[i] > 8:
                f_p = 1.6666666666666667 * numerical_flux[j][
                    i + 1] - 0.47619047619047616 * center_flux[j][
                        i + 1] + 0.11904761904761904 * numerical_flux[j][
                            i + 2] - 0.01984126984126984 * center_flux[j][
                                i +
                                2] + 0.0015873015873015873 * numerical_flux[j][
                                    i + 3]
                f_m = 1.6666666666666667 * numerical_flux[j][
                    i] - 0.47619047619047616 * center_flux[j][
                        i - 1] + 0.11904761904761904 * numerical_flux[j][
                            i - 1] - 0.01984126984126984 * center_flux[j][
                                i -
                                2] + 0.0015873015873015873 * numerical_flux[j][
                                    i - 2]
                temp[i] = (f_p - f_m) * inv_dx
            elif order_used[i] > 6:
                f_p = 1.6 * numerical_flux[j][i + 1] - 0.4 * center_flux[j][
                    i + 1] + 0.0761904761904762 * numerical_flux[j][
                        i + 2] - 0.007142857142857143 * center_flux[j][i + 2]
                f_m = 1.6 * numerical_flux[j][i] - 0.4 * center_flux[j][
                    i - 1] + 0.0761904761904762 * numerical_flux[j][
                        i - 1] - 0.007142857142857143 * center_flux[j][i - 2]
                temp[i] = (f_p - f_m) * inv_dx
            elif order_used[i] > 4:
                f_p = 1.5 * numerical_flux[j][i + 1] - 0.3 * center_flux[j][
                    i + 1] + 0.03333333333333333 * numerical_flux[j][i + 2]
                f_m = 1.5 * numerical_flux[j][i] - 0.3 * center_flux[j][
                    i - 1] + 0.03333333333333333 * numerical_flux[j][i - 1]
                temp[i] = (f_p - f_m) * inv_dx
            elif order_used[i] > 2:
                f_p = 1.3333333333333333 * numerical_flux[j][
                    i + 1] - 0.16666666666666666 * center_flux[j][i + 1]
                f_m = 1.3333333333333333 * numerical_flux[j][
                    i] - 0.16666666666666666 * center_flux[j][i - 1]
                temp[i] = (f_p - f_m) * inv_dx
            else:
                f_p = numerical_flux[j][i + 1]
                f_m = numerical_flux[j][i]
                temp[i] = (f_p - f_m) * inv_dx
        result.append(temp)
    return result


if use_numba:
    _md_impl = nb.jit(nopython=True)(_md_impl_)
    _md4_impl = nb.jit(nopython=True)(_md4_impl_)
    _md6_impl = nb.jit(nopython=True)(_md6_impl_)
    _md8_impl = nb.jit(nopython=True)(_md8_impl_)
    _md10_impl = nb.jit(nopython=True)(_md10_impl_)
    _mdv_impl = nb.jit(nopython=True)(_mdv_impl_)
    _mnd4_impl = nb.jit(nopython=True)(_mnd4_impl_)
    _mnd6_impl = nb.jit(nopython=True)(_mnd6_impl_)
    _mnd8_impl = nb.jit(nopython=True)(_mnd8_impl_)
    _mnd10_impl = nb.jit(nopython=True)(_mnd10_impl_)
    _mndv_impl = nb.jit(nopython=True)(_mndv_impl_)
else:
    print("Please install Numba for better performance.")
    _md_impl = _md_impl_
    _md4_impl = _md4_impl_
    _md6_impl = _md6_impl_
    _md8_impl = _md8_impl_
    _md10_impl = _md10_impl_
    _mdv_impl = _mdv_impl_
    _mnd4_impl = _mnd4_impl_
    _mnd6_impl = _mnd6_impl_
    _mnd8_impl = _mnd8_impl_
    _mnd10_impl = _mnd10_impl_
    _mndv_impl = _mndv_impl_

_deriv_dispatch = {
    Scheme.MD: _md_impl,
    Scheme.MD4: _md4_impl,
    Scheme.MD6: _md6_impl,
    Scheme.MD8: _md8_impl,
    Scheme.MD10: _md10_impl,
    Scheme.MDV: _mdv_impl,
    Scheme.MND4: _mnd4_impl,
    Scheme.MND6: _mnd6_impl,
    Scheme.MND8: _mnd8_impl,
    Scheme.MND10: _mnd10_impl,
    Scheme.MNDV: _mndv_impl,
}


def differentiate_flux(scheme,
                       dx,
                       numerical_fluxes,
                       center_flux=None,
                       order_used=None):
    """
    Compute the derivatives using the `scheme` and spacing `dx`.

    Applies the finite-difference scheme given by the `scheme` argument
    to all variables in the `numerical_fluxes`.


    :param Derivative.Scheme scheme: The finite-difference scheme to use.
    :param double dx: The grid spacing
    :param list numerical_fluxes: The numerical fluxes at the cell faces.
    :param list center_flux: The flux at the cell centers. Only needed
        for MND schemes.
    :param list order_used: A list of `int` at each cell indicating the
        finite-difference order to use at the cell. Normally this order
        is determined by the reconstruction scheme.
    """
    return np.asarray(_deriv_dispatch[scheme](dx, np.asarray(numerical_fluxes),
                                              center_flux, order_used))
