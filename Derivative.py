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
    MD = enum.auto()
    MD4 = enum.auto()
    MND4 = enum.auto()
    MND6 = enum.auto()
    MNDV = enum.auto()


def _md_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 2nd order finite difference using numerical fluxes
    """
    result = []
    for i in range(len(numerical_flux)):
        result.append(1.0 / dx *
                      (numerical_flux[i][1:] - numerical_flux[i][:-1]))
    return result


def _md4_impl_(dx, numerical_flux, center_flux, order_used):
    """
    Implements the 4th order finite difference using numerical fluxes
    """
    result = []
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(numerical_flux[j]) - 2):
            temp[i] = 9 / 8 * (numerical_flux[j][i + 1] -
                               numerical_flux[j][i]) / dx - 1 / 24 * (
                                   numerical_flux[j][i + 2] -
                                   numerical_flux[j][i - 1]) / dx
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
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(center_flux[j]) - 1):
            f_p = 4.0 / 3.0 * numerical_flux[j][
                i + 1] - 1 / 6 * center_flux[j][i + 1]
            f_m = 4.0 / 3.0 * numerical_flux[j][i] - 1 / 6 * center_flux[j][i -
                                                                            1]
            temp[i] = (f_p - f_m) / dx
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
    for j in range(len(numerical_flux)):
        temp = np.zeros(len(numerical_flux[j]) - 1)
        for i in range(1, len(center_flux[j]) - 2):
            f_p = 3 / 2 * numerical_flux[j][i + 1] - 3 / 10 * center_flux[j][
                i + 1] + 1 / 30 * numerical_flux[j][i + 2]
            f_m = 3 / 2 * numerical_flux[j][i] - 3 / 10 * center_flux[j][
                i - 1] + 1 / 30 * numerical_flux[j][i - 1]
            temp[i] = (f_p - f_m) / dx
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
    _mnd4_impl = nb.jit(nopython=True)(_mnd4_impl_)
    _mnd6_impl = nb.jit(nopython=True)(_mnd6_impl_)
    _mndv_impl = nb.jit(nopython=True)(_mndv_impl_)
else:
    print("Please install Numba for better performance.")
    _md_impl = _md_impl_
    _md4_impl = _md4_impl_
    _mnd4_impl = _mnd4_impl_
    _mnd6_impl = _mnd6_impl_
    _mndv_impl = _mndv_impl_

_deriv_dispatch = {
    Scheme.MD: _md_impl,
    Scheme.MD4: _md4_impl,
    Scheme.MND4: _mnd4_impl,
    Scheme.MND6: _mnd6_impl,
    Scheme.MNDV: _mndv_impl,
}


def differentiate_flux(scheme,
                       dx,
                       numerical_fluxes,
                       center_flux=None,
                       order_used=None):
    return np.asarray(_deriv_dispatch[scheme](dx, np.asarray(numerical_fluxes),
                                              center_flux, order_used))
