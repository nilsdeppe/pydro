# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np

use_numba = True
try:
    import numba as nb
except:
    use_numba = False


def _adaptive_order_9(q, i, j, recons, alpha=5.0, eps=1.0e-36):
    """
    Uses a ninth-order centered stencil for reconstruction

    .. math::
      \\begin{align}
        \\hat{q}_{i+1/2}&=\\frac{35}{32768}q_{i-4} - \\frac{45}{4096}q_{i-3}
          + \\frac{441}{8291}q_{i-2} - \\frac{735}{4096}q_{i-1} \\\\
          &+ \\frac{11025}{16384}q_{i}
          + \\frac{2205}{4096}q_{i+1} - \\frac{735}{8192}q_{i+2}
          + \\frac{63}{4096}q_{i+3} \\\\
          &- \\frac{45}{32768}q_{i+4}
      \\end{align}

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double alpha: The expected decay of increasing coefficients in
        the method.

    :param double eps: The `epsilon` parameter to ignore small values and
        impose an absolute tolerance.

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """
    c8 = -4.645463286713286 * q[j + 1] + 2.322731643356643 * q[
        j + 2] - 0.6636376123876124 * q[j + 3] + 0.08295470154845155 * q[
            j + 4] - 4.645463286713286 * q[j - 1] + 2.322731643356643 * q[
                j -
                2] - 0.6636376123876124 * q[j - 3] + 0.08295470154845155 * q[
                    j - 4] + 5.806829108391608 * q[j]

    norm_top = 0.11764705882352941 * c8**2
    norm_full = (
        q[j + 1] * (26.3541786862498 * q[j + 1] - 32.5805505335858 * q[j + 2] +
                    14.5618231193504 * q[j + 3] - 5.17421371286638 * q[j + 4] +
                    48.4889306413379 * q[j - 1] - 28.1540947588968 * q[j - 2] +
                    11.2847861700348 * q[j - 3] - 4.40079887977502 * q[j - 4] -
                    59.571198123456 * q[j]) + q[j + 2] *
        (10.7023807538414 * q[j + 2] - 9.74666025724972 * q[j + 3] +
         3.42463032646342 * q[j + 4] - 28.1540947588968 * q[j - 1] +
         15.6735229258961 * q[j - 2] - 5.83272470361744 * q[j - 3] +
         2.32442679887303 * q[j - 4] + 35.9045186964637 * q[j]) + q[j + 3] *
        (2.56005799475581 * q[j + 3] - 1.65159264752748 * q[j + 4] +
         11.2847861700348 * q[j - 1] - 5.83272470361744 * q[j - 2] +
         1.95253788421965 * q[j - 3] - 0.743793886845061 * q[j - 4] -
         15.2861815014807 * q[j]) + q[j + 4] *
        (0.49912904954453 * q[j + 4] - 4.40079887977502 * q[j - 1] +
         2.32442679887303 * q[j - 2] - 0.743793886845061 * q[j - 3] +
         0.171415816578058 * q[j - 4] + 5.66047103599009 * q[j]) + q[j - 1] *
        (26.3541786862498 * q[j - 1] - 32.5805505335858 * q[j - 2] +
         14.5618231193504 * q[j - 3] - 5.17421371286638 * q[j - 4] -
         59.571198123456 * q[j]) + q[j - 2] *
        (10.7023807538414 * q[j - 2] - 9.74666025724972 * q[j - 3] +
         3.42463032646342 * q[j - 4] + 35.9045186964637 * q[j]) + q[j - 3] *
        (2.56005799475581 * q[j - 3] - 1.65159264752748 * q[j - 4] -
         15.2861815014807 * q[j]) + q[j - 4] *
        (0.49912904954453 * q[j - 4] + 5.66047103599009 * q[j]) +
        35.444405479435 * q[j]**2)

    sensor = 0.5 * np.log10(norm_top / (norm_full + eps))
    if sensor < -alpha * np.log10(9.0):
        recons[2 * i + 1] = -0.179443359375 * q[j + 1] + 0.0538330078125 * q[
            j + 2] - 0.010986328125 * q[j + 3] + 0.001068115234375 * q[
                j + 4] + 0.538330078125 * q[j - 1] - 0.0897216796875 * q[
                    j - 2] + 0.015380859375 * q[j - 3] - 0.001373291015625 * q[
                        j - 4] + 0.67291259765625 * q[j]

        recons[2 * i + 2] = 0.538330078125 * q[j + 1] - 0.0897216796875 * q[
            j + 2] + 0.015380859375 * q[j + 3] - 0.001373291015625 * q[
                j + 4] - 0.179443359375 * q[j - 1] + 0.0538330078125 * q[
                    j - 2] - 0.010986328125 * q[j - 3] + 0.001068115234375 * q[
                        j - 4] + 0.67291259765625 * q[j]
        return True
    return False


def _adaptive_order_7(q, i, j, recons, alpha=5.0, eps=1.0e-36):
    """
    Uses a seventh-order centered stencil for reconstruction

    .. math::
      \\begin{align}
        \\hat{q}_{i+1/2}&=-\\frac{5}{1024}q_{i-3} + \\frac{21}{512}q_{i-2}
          - \\frac{175}{1024}q_{i-1} + \\frac{175}{256}q_{i} \\\\
          &+ \\frac{525}{1024}q_{i+1} - \\frac{35}{512}q_{i+2}
          + \\frac{7}{1024}q_{i+3}
      \\end{align}

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double alpha: The expected decay of increasing coefficients in
        the method.

    :param double eps: The `epsilon` parameter to ignore small values and
        impose an absolute tolerance.

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """
    norm_top = 2 * (15625 * q[j + 1] / 44352 - 3125 * q[j + 2] / 22176 +
                    3125 * q[j + 3] / 133056 + 15625 * q[j - 1] / 44352 -
                    3125 * q[j - 2] / 22176 + 3125 * q[j - 3] / 133056 -
                    15625 * q[j] / 33264)**2 / 13
    norm_full = (-395 * q[j + 1] / 6912 + 137849 * q[j + 2] / 133056 +
                 16453 * q[j + 3] / 228096 + 11875 * q[j - 1] / 532224 +
                 36727 * q[j - 2] / 133056 - 5183 * q[j - 3] / 1596672 +
                 4661 * q[j] / 7128) * (
                     24025 * q[j + 1] / 133056 + 53545 * q[j + 2] / 266112 +
                     2495 * q[j + 3] / 399168 + 24025 * q[j - 1] / 133056 +
                     53545 * q[j - 2] / 266112 + 2495 * q[j - 3] / 399168 +
                     89393 * q[j] / 399168) + norm_top
    sensor = 0.5 * np.log10(norm_top / (norm_full + eps))
    if sensor < -alpha * np.log10(7.0):
        recons[2 * i +
               1] = -175 * q[j + 1] / 1024 + 21 * q[j + 2] / 512 - 5 * q[
                   j + 3] / 1024 + 525 * q[j - 1] / 1024 - 35 * q[
                       j - 2] / 512 + 7 * q[j - 3] / 1024 + 175 * q[j] / 256

        recons[2 * i +
               2] = 525 * q[j + 1] / 1024 - 35 * q[j + 2] / 512 + 7 * q[
                   j + 3] / 1024 - 175 * q[j - 1] / 1024 + 21 * q[
                       j - 2] / 512 - 5 * q[j - 3] / 1024 + 175 * q[j] / 256
        return True
    return False


def _adaptive_order_5(q, i, j, recons, keep_positive, alpha=5.0, eps=1.0e-36):
    """
    Uses a fifth-order centered stencil for reconstruction

    .. math::
      \\hat{q}_{i+1/2}=\\frac{3}{128}q_{i-2} - \\frac{5}{32}q_{i-1}
        + \\frac{45}{64}q_{i} + \\frac{15}{32}q_{i+1} - \\frac{5}{128}q_{i+2}

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double alpha: The expected decay of increasing coefficients in
        the method.

    :param double eps: The `epsilon` parameter to ignore small values and
        impose an absolute tolerance.

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """
    norm_top = 0.2222222222222222 * (
        -1.4880952380952381 * q[j + 1] + 0.37202380952380953 * q[j + 2] -
        1.4880952380952381 * q[j - 1] + 0.37202380952380953 * q[j - 2] +
        2.232142857142857 * q[j])**2
    norm_full = (
        -0.43154761904761907 * q[j + 1] + 1.3107638888888888 * q[j + 2] +
        0.0248015873015873 * q[j - 1] + 0.24925595238095238 * q[j - 2] +
        0.8467261904761905 * q[j]) * (
            0.08680555555555555 * q[j + 1] + 0.2387152777777778 * q[j + 2] +
            0.08680555555555555 * q[j - 1] + 0.2387152777777778 * q[j - 2] +
            0.3489583333333333 * q[j]) + norm_top
    sensor = 0.5 * np.log10(norm_top / (norm_full + eps))
    if sensor < -alpha * np.log10(5.0):
        if keep_positive:
            rjm52 = (315 / 128) * q[j - 2] - (105 / 32) * q[j - 1] + (
                189 / 64) * q[j] - (45 / 32) * q[j + 1] + (35 / 128) * q[j + 2]
            rjp52 = (315 / 128) * q[j + 2] - (105 / 32) * q[j + 1] + (
                189 / 64) * q[j] - (45 / 32) * q[j - 1] + (35 / 128) * q[j - 2]
            if rjm52 <= 0.0 or rjp52 <= 0.0:
                return False

        recons[2 * i +
               1] = -0.15625 * q[j + 1] + 0.0234375 * q[j + 2] + 0.46875 * q[
                   j - 1] - 0.0390625 * q[j - 2] + 0.703125 * q[j]

        recons[2 * i +
               2] = 0.46875 * q[j + 1] - 0.0390625 * q[j + 2] - 0.15625 * q[
                   j - 1] + 0.0234375 * q[j - 2] + 0.703125 * q[j]
        if keep_positive:
            return (recons[2 * i + 1] > 0.0 and recons[2 * i + 2] > 0.0)
        return True
    return False


def _adaptive_order_3(q, i, j, recons, keep_positive, alpha=3.0, eps=1.0e-36):
    """
    Uses a third-order centered stencil for reconstruction

    .. math::
      \\hat{q}_{i+1/2}=\\frac{1}{8}q_{i-1} + \\frac{3}{4}q_{i} +
        \\frac{3}{8}q_{i+1}

    How oscillatory the resulting polynomial is can be determined by
    comparing

    .. math::
      s^j_N = \\frac{1}{2}\\log_{10}\\left(\\frac{\\bar{\\kappa}_N}
        {\\kappa_N + \\epsilon}\\right),

    where

    .. math::
      \\begin{align}
        \\bar{\\kappa}_3 &= \\frac{2}{5}\\left(\\frac{3}{4}q_{i+1} -
          \\frac{3}{2}q_{i} + \\frac{3}{4}q_{i-1}\\right)^{2}, \\\\
      \\kappa_3 &=  \\left(\\frac{3}{8}q_{i+1} + \\frac{1}{4}q_{i} +
             \\frac{3}{8}q_{i-1}\\right)
             \\left(\\frac{31}{20}q_{i+1} - \\frac{1}{10}q_{i} +
             \\frac{11}{20}q_{i-1}\\right),
      \\end{align}

    to

    .. math::
      -\\alpha\\log_{10}(4)

    Typically :math:`\\alpha\\sim4` so that the coefficients decay
    as :math:`1/3^4`.

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double alpha: The expected decay of increasing coefficients in
        the method.

    :param double eps: The `epsilon` parameter to ignore small values and
        impose an absolute tolerance.

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """
    norm_top = 0.4 * (0.74 * q[j + 1] - 1.5 * q[j] + 0.75 * q[j - 1])**2
    norm_full = (0.375 * q[j + 1] + 0.25 * q[j] + 0.375 * q[j - 1]) * (
        1.55 * q[j + 1] - 0.1 * q[j] + 0.55 * q[j - 1])
    sensor = 0.5 * np.log10(norm_top / (norm_full + eps))
    if sensor < -alpha * np.log10(alpha):
        if keep_positive:
            rjm32 = 1.875 * q[j - 1] - 1.25 * q[j] - 0.125 * q[j + 1]
            rjp32 = 1.875 * q[j + 1] - 1.25 * q[j] - 0.125 * q[j - 1]
            if rjm32 <= 0.0 or rjp32 <= 0.0:
                return False

        recons[2 * i + 1] = 0.375 * q[j - 1] + 0.75 * q[j] - 0.125 * q[j + 1]

        recons[2 * i + 2] = -0.125 * q[j - 1] + 0.75 * q[j] + 0.375 * q[j + 1]
        if keep_positive:
            return (recons[2 * i + 1] > 0.0 and recons[2 * i + 2] > 0.0)
        return True
    return False


def _adaptive_order_weno3_robust(q,
                                 i,
                                 j,
                                 recons,
                                 keep_positive,
                                 eps=1.0e-45,
                                 c1=1.0,
                                 c2=1.0e5,
                                 c3=1.0,
                                 exponent=1.0,
                                 wenoz=False):
    """
    A robust WENO3 reconstruction using 5 points.

    The individual polynomials stencils for the reconstruction are written as

    .. math::
      \\begin{align}
        u(\\xi) = u_0 + u_\\xi P_1(\\xi) + u_{\\xi\\xi} P_2(\\xi)
      \\end{align}

    The left-, central-, and right-biased stencils for the one-dimensional
    coefficients are:

    .. math::
      \\begin{align}
        u_{\\xi}^{(L)}&=\\frac{1}{2} u_{-2} - 2u_{-1} + \\frac{3}{2} u_0 \\\\
        u_{\\xi\\xi}^{(L)}&=\\frac{u_{-2} - 2 u_{-1} + u_0}{2} \\\\
        u_{\\xi}^{(C)}&=\\frac{1}{2}(u_1 - u_{-1}) \\\\
        u_{\\xi\\xi}^{(C)}&=\\frac{u_{-1} - 2 u_0 + u_1}{2} \\\\
        u_{\\xi}^{(R)}&=-\\frac{3}{2}u_0 + 2 u_1 - \\frac{1}{2} u_2 \\\\
        u_{\\xi\\xi}^{(R)}&=\\frac{u_{0} - 2 u_{1} + u_2}{2}.
      \\end{align}

    The oscillation indicators are given by

    .. math::
      \\beta_{(i)} = \\left(u_\\xi^{(i)}\\right)^2 +
        \\frac{13}{3}\\left(u_{\\xi\\xi}^{(i)}\\right)^2,

    where :math:`i\\in\{L,C,R\}`. The nonlinear weights are:

    .. math::
      \\begin{align}
        \\omega_k &= \\frac{\\alpha_k}{\sum_{l=0}^{2}\\alpha_l} \\\\
        \\alpha_k &= \\frac{\\lambda_k}{(\\beta_k + \\epsilon_k)^p}
      \\end{align}

    where :math:`p` is usually chosen to be 4 or 8, and :math:`\\lambda_0=1`,
    :math:`\\lambda_1=10^5`, and :math:`\\lambda_2=1`.

    In the case where the WENOZ weights are used :math:`p=1` and the
    new oscillation indicators are

    .. math::
      \\beta_k^Z=\\frac{\\beta_k}{\\beta_k + \\tau_5 + \\epsilon_k}

    where

    .. math::
      \\tau_5 = |\\beta_3 - \\beta_1|.

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double eps: The `epsilon` parameter to avoid division by zero.

    :param double c0: The linear weight :math:`\\lambda_{0}`.

    :param double c1: The linear weight :math:`\\lambda_{1}`.

    :param double c2: The linear weight :math:`\\lambda_{2}`.

    :param double exponent: The exponent :math:`p` in denominator of the
        :math:`\\alpha_k`. Set to 1 when using WENOZ weights.

    :param bool wenoz: If `True` then use the WENOZ weights.

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """

    if wenoz:
        exponent = 1

    s1_ux = -2.0 * q[j - 1] + 0.5 * q[j - 2] + 1.5 * q[j]
    s1_ux2 = 0.5 * (q[j - 2] - 2.0 * q[j - 1] + q[j])
    s2_ux = 0.5 * (q[j + 1] - q[j - 1])
    s2_ux2 = 0.5 * (q[j - 1] - 2.0 * q[j] + q[j + 1])
    s3_ux = -1.5 * q[j] + 2.0 * q[j + 1] - 0.5 * (q[j + 2])
    s3_ux2 = (q[j] - 2.0 * q[j + 1] + q[j + 2])
    beta1 = s1_ux**2 + (13.0 / 3.0) * s1_ux2**2
    beta2 = s2_ux**2 + (13.0 / 3.0) * s2_ux2**2
    beta3 = s3_ux**2 + (13.0 / 3.0) * s3_ux2**2

    if wenoz:
        # WENOZ
        tau5 = np.abs(beta3 - beta1)
        beta1 = beta1 / (beta1 + tau5 + eps)
        beta2 = beta2 / (beta2 + tau5 + eps)
        beta3 = beta3 / (beta3 + tau5 + eps)

    alpha_denom1 = (eps + beta1)**exponent
    alpha_denom2 = (eps + beta2)**exponent
    alpha_denom3 = (eps + beta3)**exponent

    alpha1 = c1 / alpha_denom1
    alpha2 = c2 / alpha_denom2
    alpha3 = c3 / alpha_denom3

    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    # L0 = 1
    # L1(1/2) = 1/2    L1(-1/2) = -1/2
    # L2(1/2) = 1/6    L2(-1/2) = 1/6
    if keep_positive:

        def sgn(x):
            return -1.0 if x < 0.0 else (1.0 if x > 0.0 else 0.0)

        # a x^2 + b x + c
        a = w1 * s1_ux2 + w2 * s2_ux2 + w3 * s3_ux2
        b = w1 * s1_ux + w2 * s2_ux + w3 * s3_ux
        c = (w1 + w2 + w3) * q[j] - 1.0 / 12.0 * (w1 * s1_ux2 + w2 * s2_ux2 +
                                                  w3 * s3_ux2)

        q_root = -0.5 * (b + sgn(b) * np.sqrt(b**2 - 4.0 * a * c))
        x1 = q_root / a
        x2 = c / q_root
        # Have a negative root, so return False
        root_bound = 2.5
        if (x1 < root_bound and x1 > -root_bound) or (x2 < root_bound
                                                      and x2 > -root_bound):
            return False

    recons[
        2 * i +
        1] = w1 * (q[j] + s1_ux * (-0.5) + s1_ux2 *
                   (1.0 / 6.0)) + w2 * (q[j] + s2_ux * (-0.5) + s2_ux2 *
                                        (1.0 / 6.0)) + w3 * (q[j] + s3_ux *
                                                             (-0.5) + s3_ux2 *
                                                             (1.0 / 6.0))

    recons[
        2 * i +
        2] = w1 * (q[j] + s1_ux * (0.5) + s1_ux2 *
                   (1.0 / 6.0)) + w2 * (q[j] + s2_ux * (0.5) + s2_ux2 *
                                        (1.0 / 6.0)) + w3 * (q[j] + s3_ux *
                                                             (0.5) + s3_ux2 *
                                                             (1.0 / 6.0))
    return True


def _adaptive_order_wcns3z(q,
                           i,
                           j,
                           recons,
                           keep_positive,
                           eps=1.0e-17,
                           c0=0.25,
                           c1=0.75,
                           liu_indicators=True):
    """
    A general third order weight compact nonlinear scheme using the Z weights.

    The same as :py:meth:`adaptive_order_wcns3()` except that the Z
    weights are used. First we define

    .. math::
      \\tau_3 = |\\beta_1 - \\beta_0|

    Then the new :math:`\\alpha_k` are given by

    .. math::
      \\alpha_k = c_k\\left(1 + \\frac{\\tau_3}
        {\\beta_k + \\epsilon_k}\\right),

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double eps: The `epsilon` parameter to avoid division by zero.

    :param double c0: The optimal linear weight :math:`c_{0}`. For 3rd
        order use :math:`1/4`. For 2nd order but increased robustness use
        :math:`1/2`.

    :param double c1: The optimal linear weight :math:`c_{1}`. For 3rd
        order use :math:`3/4`. For 2nd order but increased robustness use
        :math:`1/2`.

    :param bool liu_indicators: If `True` use the oscillation indicators
        of :cite:`Liu2018`

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """
    if not liu_indicators:
        beta0 = (q[j] - q[j - 1])**2
        beta1 = (q[j + 1] - q[j])**2
    else:
        beta0 = 0.25 * (np.abs(q[j + 1] - q[j - 1]) -
                        np.abs(4.0 * q[j] - 3.0 * q[j - 1] - q[j + 1]))**2
        beta1 = 0.25 * (np.abs(q[j + 1] - q[j - 1]) -
                        np.abs(3.0 * q[j + 1] + q[j - 1] - 4.0 * q[j]))**2

    tau3 = np.abs(beta1 - beta0)

    alpha_l0 = c1 * (1.0 + tau3 / (beta0 + eps *
                                   (1 + abs(q[j]) + abs(q[j - 1]))))
    alpha_l1 = c0 * (1.0 + tau3 / (beta1 + eps *
                                   (1 + abs(q[j]) + abs(q[j + 1]))))
    w_l0 = alpha_l0 / (alpha_l0 + alpha_l1)
    w_l1 = alpha_l1 / (alpha_l0 + alpha_l1)
    if keep_positive:
        denom = (w_l0 * (q[j] - q[j - 1]) + w_l1 * (q[j + 1] - q[j]))
        numer = -(w_l0 + w_l1) * q[j]
        if denom != 0.0 and np.abs(numer / denom) < 2.0:
            return False

    recons[2 * i +
           1] = w_l0 * (0.5 * q[j] + 0.5 * q[j - 1]) + w_l1 * (1.5 * q[j] -
                                                               0.5 * q[j + 1])

    alpha_r0 = c0 * (1.0 + tau3 / (beta0 + eps *
                                   (1 + abs(q[j]) + abs(q[j - 1]))))
    alpha_r1 = c1 * (1.0 + tau3 / (beta1 + eps *
                                   (1 + abs(q[j]) + abs(q[j + 1]))))
    w_r0 = alpha_r0 / (alpha_r0 + alpha_r1)
    w_r1 = alpha_r1 / (alpha_r0 + alpha_r1)

    if keep_positive:
        denom = (w_r0 * (q[j] - q[j - 1]) + w_r1 * (q[j + 1] - q[j]))
        numer = -(w_r0 + w_r1) * q[j]
        if denom != 0.0 and np.abs(numer / denom) < 2.0:
            return False

    recons[2 * i +
           2] = w_r0 * (1.5 * q[j] - 0.5 * q[j - 1]) + w_r1 * (0.5 * q[j] +
                                                               0.5 * q[j + 1])
    return True


def _adaptive_order_wcns3(q,
                          i,
                          j,
                          recons,
                          keep_positive,
                          eps=1.0e-17,
                          c0=0.25,
                          c1=0.75,
                          liu_indicators=True,
                          exponent=2):
    """
    A general third order weight compact nonlinear scheme.

    Third order WCNS3 reconstruction is done by first defining
    oscillation indicators :math:`\\beta_0` and :math:`\\beta_1` as

    .. math::
      \\begin{align}
        \\beta_0 &= (q_i - q_{i-1})^2 \\\\
        \\beta_1 &= (q_{i+1} - q_{i})^2
      \\end{align}

    We refer to these as the standard oscillation indicators, but also
    provide the improved oscillation indicators of Liu :cite:`Liu2018`:

    .. math::
      \\begin{align}
        \\beta_0 &= \\frac{1}{4}\\left(\\lvert q_{i+1} - q_{i-1}\\rvert
            - \\lvert4 q_i - 3 q_{i-1} - q_{i+1}\\rvert\\right)^2, \\\\
        \\beta_1 &= \\frac{1}{4}\\left(\\lvert q_{i+1} - q_{i-1}\\rvert
            - \\lvert3 q_{i+1} + q_{i-1} - 4 q_{i}\\rvert\\right)^2.
      \\end{align}

    Then coefficients :math:`\\alpha_k` are defined as

    .. math::
      \\alpha_k = \\frac{c_k}{(\\beta_k + \\epsilon_k)^p}

    where :math:`\\epsilon_k` is a factor used to avoid division
    by zero and is set to

    .. math::
      \\begin{align}
        \\epsilon_0 &= \\epsilon\\left(1 + |q_{i}| + |q_{i-1}|\\right) \\\\
        \\epsilon_1 &= \\epsilon\\left(1 + |q_{i}| + |q_{i+1}|\\right)
      \\end{align}

    and the linear weights are :math:`c_0=1/4` and :math:`c_1=3/4`.
    Finally, we define the nonlinear weights:

    .. math::
      \\omega_k=\\frac{\\alpha_k}{\sum_{k=0}^{1}\\alpha_k}


    The reconstruction stencils are given by:

    .. math::
      \\begin{align}
         q^0_{i+1/2}&= \\frac{3}{2}q_{i} - \\frac{1}{2}q_{i-1}, \\\\
         q^1_{i+1/2}&=\\frac{1}{2}q_i + \\frac{1}{2}q_{i+1},
      \\end{align}

    The final reconstructed solution is given by

    .. math::
      \hat{q}_{i+1/2}=\sum_{k=0}^{1}\omega_k q^k_{i+1/2}

    :param q: The variable values at the cell centers.
    :type q: list of double

    :param int i: The index into the reconstructed array

    :param int j: The index of the cell whose faces are being
        reconstructed in `q`

    :param recons: The array of the reconstructed variable.
    :type recons: list of double

    :param bool keep_positive: If `True` then returns `False` if the
        reconstructed solution is not positive.

    :param double eps: The `epsilon` parameter to avoid division by zero.

    :param double c0: The optimal linear weight :math:`c_{0}`. For 3rd
        order use :math:`1/4`. For 2nd order but increased robustness use
        :math:`1/2`.

    :param double c1: The optimal linear weight :math:`c_{1}`. For 3rd
        order use :math:`3/4`. For 2nd order but increased robustness use
        :math:`1/2`.

    :param bool liu_indicators: If `True` use the oscillation indicators
        of :cite:`Liu2018`

    :param int exponent: The exponent :math:`p` in denominator of the
        :math:`\\alpha_k`

    :return: (`bool`) `True` if the reconstruction was successful, otherwise
        `False`
    """
    if not liu_indicators:
        beta0 = (q[j] - q[j - 1])**2
        beta1 = (q[j + 1] - q[j])**2
    else:
        beta0 = 0.25 * (np.abs(q[j + 1] - q[j - 1]) -
                        np.abs(4.0 * q[j] - 3.0 * q[j - 1] - q[j + 1]))**2
        beta1 = 0.25 * (np.abs(q[j + 1] - q[j - 1]) -
                        np.abs(3.0 * q[j + 1] + q[j - 1] - 4.0 * q[j]))**2

    alpha_denom0 = (eps * (1 + abs(q[j]) + abs(q[j - 1])) + beta0)**exponent
    alpha_denom1 = (eps * (1 + abs(q[j + 1]) + abs(q[j])) + beta1)**exponent

    alpha_l0 = c1 / alpha_denom0
    alpha_l1 = c0 / alpha_denom1
    w_l0 = alpha_l0 / (alpha_l0 + alpha_l1)
    w_l1 = alpha_l1 / (alpha_l0 + alpha_l1)
    if keep_positive:
        denom = (w_l0 * (q[j] - q[j - 1]) + w_l1 * (q[j + 1] - q[j]))
        numer = -(w_l0 + w_l1) * q[j]
        if denom != 0.0 and np.abs(numer / denom) < 2.0:
            return False

    recons[2 * i +
           1] = w_l0 * (0.5 * q[j] + 0.5 * q[j - 1]) + w_l1 * (1.5 * q[j] -
                                                               0.5 * q[j + 1])

    alpha_r0 = c0 / alpha_denom0
    alpha_r1 = c1 / alpha_denom1
    w_r0 = alpha_r0 / (alpha_r0 + alpha_r1)
    w_r1 = alpha_r1 / (alpha_r0 + alpha_r1)

    if keep_positive:
        denom = (w_r0 * (q[j] - q[j - 1]) + w_r1 * (q[j + 1] - q[j]))
        numer = -(w_r0 + w_r1) * q[j]
        if denom != 0.0 and np.abs(numer / denom) < 2.0:
            return False

    recons[2 * i +
           2] = w_r0 * (1.5 * q[j] - 0.5 * q[j - 1]) + w_r1 * (0.5 * q[j] +
                                                               0.5 * q[j + 1])
    return True


def _adaptive_order_1(q, i, j, recons):
    """
    First-order reconstruction.

    First-order reconstruction is given by

    .. math::
      \hat{q}_{i + 1/2} = q_i

    """
    recons[2 * i + 1] = q[j]
    recons[2 * i + 2] = q[j]
    return True


if use_numba:
    adaptive_order_9 = nb.njit(_adaptive_order_9)
    adaptive_order_7 = nb.njit(_adaptive_order_7)
    adaptive_order_5 = nb.njit(_adaptive_order_5)
    adaptive_order_3 = nb.njit(_adaptive_order_3)
    adaptive_order_wcns3 = nb.njit(_adaptive_order_wcns3)
    adaptive_order_wcns3z = nb.njit(_adaptive_order_wcns3z)
    adaptive_order_weno3_robust = nb.njit(_adaptive_order_weno3_robust)
    adaptive_order_1 = nb.njit(_adaptive_order_1)
else:
    adaptive_order_9 = _adaptive_order_9
    adaptive_order_7 = _adaptive_order_7
    adaptive_order_5 = _adaptive_order_5
    adaptive_order_3 = _adaptive_order_3
    adaptive_order_wcns3 = _adaptive_order_wcns3
    adaptive_order_wcns3z = _adaptive_order_wcns3z
    adaptive_order_weno3_robust = _adaptive_order_weno3_robust
    adaptive_order_1 = _adaptive_order_1
