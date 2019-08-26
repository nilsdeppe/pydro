# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np


def adams_bashforth(order, history, evolved_vars, dvars_dt, dt):
    """
    A simple 1st, 2nd, 3rd, or 4th order explicit Adams-Bashforth
    time stepper.
    """
    history.append(dvars_dt)

    if len(history) > order:
        history.pop(0)

    if len(history) == 1:
        coeffs = np.asarray([1.0])
    elif len(history) == 2:
        coeffs = np.asarray([-1.0 / 2.0, 3.0 / 2.0])
    elif len(history) == 3:
        coeffs = np.asarray([5.0 / 12.0, -16.0 / 12.0, 23.0 / 12.0])
    elif len(history) == 4:
        coeffs = np.asarray(
            [-9.0 / 24.0, 37.0 / 24.0, -59.0 / 24.0, 55.0 / 24.0])
    else:
        raise ValueError("Order must be 1, 2, 3, or 4, not %d" % order)

    for i in range(len(history)):
        for j in range(len(dvars_dt)):
            evolved_vars[j] += dt * coeffs[i] * history[i][j]
    return evolved_vars
