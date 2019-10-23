# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np

import Derivative


class Rk3Ssp:
    _compute_time_deriv = None
    _evolved_vars = None
    _time = None
    _cfl_coefficient = 1.0

    def __init__(self, time_deriv, initial_state, initial_time):
        self._compute_time_deriv = time_deriv
        self._evolved_vars = np.copy(initial_state)
        self._time = initial_time

    def get_evolved_vars(self):
        return self._evolved_vars

    def get_time(self):
        return self._time

    def get_cfl_coefficient(self):
        return self._cfl_coefficient

    def take_step(self, dt):
        dt_vars = self._compute_time_deriv(self._evolved_vars, self._time)
        v1 = self._evolved_vars + dt * dt_vars
        dt_vars = dt_vars = self._compute_time_deriv(v1, self._time + dt)
        v2 = 0.25 * (3.0 * self._evolved_vars + v1 + dt * dt_vars)
        dt_vars = self._compute_time_deriv(v2, self._time + 0.5 * dt)
        self._evolved_vars = 1.0 / 3.0 * (self._evolved_vars + 2.0 * v2 +
                                          2.0 * dt * dt_vars)
        self._time += dt


class Rk4Ssp:
    """
    A fourth-order SSP Runge-Kutta method from
    Strong Stability Preserving High Order Time Discretization Methods
    by Gottlieb, Shu, and Tadmor
    """
    _compute_time_deriv = None
    _evolved_vars = None
    _time = None
    _cfl_coefficient = 1.0

    def __init__(self, time_deriv, initial_state, initial_time):
        self._compute_time_deriv = time_deriv
        self._evolved_vars = np.copy(initial_state)
        self._time = initial_time

    def get_evolved_vars(self):
        return self._evolved_vars

    def get_time(self):
        return self._time

    def get_cfl_coefficient(self):
        return self._cfl_coefficient

    def take_step(self, dt):
        k0 = self._compute_time_deriv(self._evolved_vars, self._time)

        v1 = self._evolved_vars + dt * k0
        k1 = self._compute_time_deriv(v1, self._time)

        v2 = v1 + dt * k1
        k2 = self._compute_time_deriv(v2, self._time)

        v3 = v2 + dt * k2
        k3 = self._compute_time_deriv(v3, self._time)

        self._evolved_vars = (0.375 * self._evolved_vars +
                              0.333333333333333333 * v1 + 0.25 * v2 +
                              0.041666666666666664 * (v3 + dt * k3))

        self._time += dt


class Rk6Ssp:
    """
    A sixth-order SSP Runge-Kutta method from
    Strong Stability Preserving High Order Time Discretization Methods
    by Gottlieb, Shu, and Tadmor
    """
    _compute_time_deriv = None
    _evolved_vars = None
    _time = None
    _cfl_coefficient = 1.0

    def __init__(self, time_deriv, initial_state, initial_time):
        self._compute_time_deriv = time_deriv
        self._evolved_vars = np.copy(initial_state)
        self._time = initial_time

    def get_evolved_vars(self):
        return self._evolved_vars

    def get_time(self):
        return self._time

    def get_cfl_coefficient(self):
        return self._cfl_coefficient

    def take_step(self, dt):
        k0 = self._compute_time_deriv(self._evolved_vars, self._time)

        v1 = self._evolved_vars + dt * k0
        k1 = self._compute_time_deriv(v1, self._time)

        v2 = v1 + dt * k1
        k2 = self._compute_time_deriv(v2, self._time)

        v3 = v2 + dt * k2
        k3 = self._compute_time_deriv(v3, self._time)

        v4 = v3 + dt * k3
        k4 = self._compute_time_deriv(v4, self._time)

        v5 = v4 + dt * k4
        k5 = self._compute_time_deriv(v5, self._time)

        self._evolved_vars = (0.3680555555555556 * self._evolved_vars +
                              0.36666666666666664 * v1 + 0.1875 * v2 +
                              0.05555555555555555 * v3 +
                              0.020833333333333332 * v4 +
                              0.001388888888888889 * (v5 + dt * k5))

        self._time += dt


class Rk8Ssp:
    """
    A eighth-order SSP Runge-Kutta method from
    Strong Stability Preserving High Order Time Discretization Methods
    by Gottlieb, Shu, and Tadmor
    """
    _compute_time_deriv = None
    _evolved_vars = None
    _time = None
    _cfl_coefficient = 1.0

    def __init__(self, time_deriv, initial_state, initial_time):
        self._compute_time_deriv = time_deriv
        self._evolved_vars = np.copy(initial_state)
        self._time = initial_time

    def get_evolved_vars(self):
        return self._evolved_vars

    def get_time(self):
        return self._time

    def get_cfl_coefficient(self):
        return self._cfl_coefficient

    def take_step(self, dt):
        k0 = self._compute_time_deriv(self._evolved_vars, self._time)

        v1 = self._evolved_vars + dt * k0
        k1 = self._compute_time_deriv(v1, None)

        v2 = v1 + dt * k1
        k2 = self._compute_time_deriv(v2, None)

        v3 = v2 + dt * k2
        k3 = self._compute_time_deriv(v3, None)

        v4 = v3 + dt * k3
        k4 = self._compute_time_deriv(v4, None)

        v5 = v4 + dt * k4
        k5 = self._compute_time_deriv(v5, None)

        v5 = v4 + dt * k4
        k5 = self._compute_time_deriv(v5, None)

        v6 = v5 + dt * k5
        k6 = self._compute_time_deriv(v6, None)

        v7 = v6 + dt * k6
        k7 = self._compute_time_deriv(v7, None)

        self._evolved_vars = (0.36788194444444444 * self._evolved_vars +
                              0.3678571428571429 * v1 +
                              0.1840277777777778 * v2 +
                              0.06111111111111111 * v3 + 0.015625 * v4 +
                              0.002777777777777778 * v5 +
                              0.0006944444444444445 * v6 +
                              2.48015873015873e-05 * (v7 + dt * k7))

        self._time += dt


def _adams_bashforth1(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * history[0][j]


def _adams_bashforth2(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * (-0.5 * history[0][j] + 1.5 * history[1][j])


def _adams_bashforth3(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * (0.4166666666666667 * history[0][j] -
                                 1.3333333333333333 * history[1][j] +
                                 1.9166666666666667 * history[2][j])


def _adams_bashforth4(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * (-0.375 * history[0][j] +
                                 1.5416666666666667 * history[1][j] -
                                 2.4583333333333335 * history[2][j] +
                                 2.2916666666666665 * history[3][j])


def _adams_bashforth(order, history, evolved_vars, dvars_dt, dt):
    """
    A simple 1st, 2nd, 3rd, or 4th order explicit Adams-Bashforth
    time stepper.
    """
    history.append(dvars_dt)

    if len(history) > order:
        history.pop(0)

    if len(history) == 1:
        _adams_bashforth1(history, evolved_vars, dvars_dt, dt)
    elif len(history) == 2:
        _adams_bashforth2(history, evolved_vars, dvars_dt, dt)
    elif len(history) == 3:
        _adams_bashforth3(history, evolved_vars, dvars_dt, dt)
    elif len(history) == 4:
        _adams_bashforth4(history, evolved_vars, dvars_dt, dt)
    else:
        raise ValueError("Order must be 1, 2, 3, or 4, not %d" % order)


class AdamsBashforth:
    _evolved_vars = None
    _primitive_vars = None
    _recons_evolved_vars = None
    _recons_primitive_vars = None
    _compute_numerical_flux = None
    _compute_flux = None
    _compute_sources = None
    _compute_primitive_vars = None
    _compute_evolved_vars = None
    _time = 0.0
    _x = None
    _dx = None
    _reconstruction_scheme = None
    _deriv_scheme = None
    _order_used = None
    _reconstruct_primitives = None
    _reconstructor = None

    # Time stepping specific variables
    _history = []
    _order = 3

    def __init__(self, order, reconstruction_scheme, deriv_scheme,
                 initial_state, initial_time, compute_primitive_vars,
                 compute_evolved_vars, compute_flux, compute_sources,
                 compute_numerical_flux, x, reconstruct_primitives,
                 reconstructor):
        print("Unsupported time stepper AdamsBashforth.")
        import sys
        sys.exit(1)
        self._order = order
        self._reconstruction_scheme = reconstruction_scheme
        self._deriv_scheme = deriv_scheme
        self._evolved_vars = np.copy(initial_state)
        self._time = initial_time
        # System specific
        self._compute_primitive_vars = compute_primitive_vars
        self._compute_evolved_vars = compute_evolved_vars
        self._compute_flux = compute_flux
        self._compute_sources = compute_sources
        self._compute_numerical_flux = compute_numerical_flux
        # Grid and reconstruction
        self._x = np.copy(x)
        self._dx = self._x[1] - self._x[0]
        self._order_used = np.zeros(len(x), dtype=int)
        self._history = []
        self._reconstruct_primitives = reconstruct_primitives
        self._reconstructor = reconstructor
        print("Self dx: ", self._dx)

    def get_evolved_vars(self):
        return self._evolved_vars

    def get_time(self):
        return self._time

    def get_order_used(self):
        return self._order_used

    def _reset_order_used(self, length):
        if len(self._order_used) != length:
            self._order_used = np.zeros(length, dtype=int) + 100
        else:
            self._order_used[:] = 100

    def _reconstruct_variables(self):
        self._reset_order_used(len(self._evolved_vars[0]))

        if self._reconstruct_primitives:
            self._primitive_vars = self._compute_primitive_vars(
                self._primitive_vars, self._evolved_vars)
            self._recons_primitive_vars = self._reconstructor(
                self._primitive_vars, self._reconstruction_scheme,
                self._order_used)
            self._recons_evolved_vars = self._compute_evolved_vars(
                self._recons_primitive_vars)
        else:
            self._recons_evolved_vars = self._reconstructor(
                self._evolved_vars, self._reconstruction_scheme,
                self._order_used)

    def take_step(self, dt):
        self._reconstruct_variables()

        numerical_fluxes_at_faces = self._compute_numerical_flux(
            self._recons_evolved_vars)

        dt_evolved_vars = -1.0 * Derivative.differentiate_flux(
            self._deriv_scheme, self._dx, numerical_fluxes_at_faces,
            self._compute_flux(self._evolved_vars)
            if self._deriv_scheme == Derivative.Scheme.MND4
            or self._deriv_scheme == Derivative.Scheme.MND6
            or self._deriv_scheme == Derivative.Scheme.MNDV else None,
            self._order_used)

        bc_distance = 7
        # zero time derivs at boundary
        for i in range(len(dt_evolved_vars)):
            dt_evolved_vars[i][0:bc_distance] = 0.0
            dt_evolved_vars[i][-bc_distance:] = 0.0

        _adams_bashforth(self._order, self._history, self._evolved_vars,
                         dt_evolved_vars, dt)

        self._time += dt
