# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np


def _sedov_calc_a(gamma, nu):
    """
    Compute the exponents of the Sedov solution.
    nu = 1 - planar
    nu = 2 - cylindrical
    nu = 3 - spherical
    """
    a = [None] * 8

    a[0] = 2.0 / (nu + 2.0)
    a[2] = (1.0 - gamma) / (2.0 * (gamma - 1.0) + nu)
    a[3] = nu / (2.0 * (gamma - 1.0) + nu)
    a[5] = 2.0 / (gamma - 2.0)
    a[6] = gamma / (2.0 * (gamma - 1.0) + nu)

    a[1] = (((nu + 2.0) * gamma) /
            (2.0 + nu * (gamma - 1.0))) * ((2.0 * nu * (2.0 - gamma)) /
                                           (gamma * (nu + 2.0)**2) - a[2])
    a[4] = a[1] * (nu + 2.0) / (2.0 - gamma)
    a[7] = (2.0 + nu * (gamma - 1.0)) * a[1] / (nu * (2.0 - gamma))
    return a


def _sedov_calc_beta(v, gamma, nu):
    """
    Compute the beta values for the sedov solution (coefficients
    of the polynomials of the similarity variables)
    v - the similarity variable
    nu = 1 - planar
    nu = 2 - cylindrical
    nu = 3 - spherical
    """

    beta = (nu + 2.0) * (gamma + 1.0) * np.array(
        (0.25, (gamma / (gamma - 1)) * 0.5, -(2.0 + nu * (gamma - 1.0)) / 2.0 /
         ((nu + 2.0) * (gamma + 1.0) - 2.0 *
          (2.0 + nu * (gamma - 1.0))), -0.5 / (gamma - 1.0)))

    beta = np.outer(beta, v)

    beta += (gamma + 1.0) * np.array(
        (0.0, -1.0 / (gamma - 1.0), (nu + 2.0) /
         ((nu + 2.0) * (gamma + 1.0) - 2.0 *
          (2.0 + nu * (gamma - 1.0))), 1.0 / (gamma - 1.0))).reshape((4, 1))

    return beta


def sedov(t, E0, rho0, gamma, num_points=1000, nu=3):
    """
    Solve the sedov problem on the interval [0, shock radius]
    t - the time
    E0 - the initial energy
    rho0 - the initial density
    n - number of points used for evaluating integrals (1000)
    nu - the dimension/symmetry, 1 = planar, 2 = cylindrical, 3 = spherical
    g - the polytropic gas gamma

    Returns the radius, pressure, density, velocity, shock radius,
    pressure at shock, density at shock, velocity at shock, and
    the shock speed.
    """
    from scipy.special import gamma as Gamma

    # the similarity variable
    v_min = 2.0 / ((nu + 2) * gamma)
    v_max = 4.0 / ((nu + 2) * (gamma + 1))

    v = v_min + np.arange(num_points) * (v_max - v_min) / (num_points - 1.0)

    a = _sedov_calc_a(gamma, nu)
    beta = _sedov_calc_beta(v, gamma, nu)
    lbeta = np.log(beta)

    r = np.exp(-a[0] * lbeta[0] - a[2] * lbeta[1] - a[1] * lbeta[2])
    rho = ((gamma + 1.0) /
           (gamma - 1.0)) * np.exp(a[3] * lbeta[1] + a[5] * lbeta[3] +
                                   a[4] * lbeta[2])
    p = np.exp(nu * a[0] * lbeta[0] + (a[5] + 1) * lbeta[3] +
               (a[4] - 2 * a[1]) * lbeta[2])
    u = beta[0] * r * 4.0 / ((gamma + 1.0) * (nu + 2.0))
    p *= 8.0 / ((gamma + 1.0) * (nu + 2.0) * (nu + 2.0))

    # we have to take extra care at v=v_min, since this can be a special point.
    # It is not a singularity, however, the gradients of our variables (wrt v)
    # are:
    # r -> 0, u -> 0, rho -> 0, p-> constant

    u[0] = 0.0
    rho[0] = 0.0
    r[0] = 0.0
    p[0] = p[1]

    # volume of an n-sphere
    vol = (np.pi**(nu / 2.0) / Gamma(nu / 2.0 + 1.0)) * np.power(r, nu)

    # note we choose to evaluate the integral in this way because the
    # volumes of the first few elements (i.e near v=vmin) are shrinking
    # very slowly, so we dramatically improve the error convergence by
    # finding the volumes exactly. This is most important for the
    # pressure integral, as this is on the order of the volume.

    # (dimensionless) energy of the model solution
    de = rho * u * u * 0.5 + p / (gamma - 1.0)
    # integrate (trapezium rule)
    q = np.inner(de[1:] + de[:-1], np.diff(vol)) * 0.5

    # the factor to convert to this particular problem
    fac = (q * (t**nu) * rho0 / E0)**(-1.0 / (nu + 2.0))

    # shock speed
    shock_speed = fac * (2.0 / (nu + 2.0))
    rho_s = ((gamma + 1.0) / (gamma - 1.0)) * rho0
    r_s = shock_speed * t * (nu + 2.0) / 2.0
    p_s = (2.0 * rho0 * shock_speed * shock_speed) / (gamma + 1.0)
    u_s = (2.0 * shock_speed) / (gamma + 1.0)

    r *= fac * t
    u *= fac
    p *= fac * fac * rho0
    rho *= rho0

    return r, p, rho, u, r_s, p_s, rho_s, u_s, shock_speed
