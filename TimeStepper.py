# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)


def _adams_bashforth1(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * history[0][j]
    return evolved_vars


def _adams_bashforth2(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * (-0.5 * history[0][j] + 1.5 * history[1][j])
    return evolved_vars


def _adams_bashforth3(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * (0.4166666666666667 * history[0][j] -
                                 1.3333333333333333 * history[1][j] +
                                 1.9166666666666667 * history[2][j])
    return evolved_vars


def _adams_bashforth4(history, evolved_vars, dvars_dt, dt):
    for j in range(len(dvars_dt)):
        evolved_vars[j] += dt * (-0.375 * history[0][j] +
                                 1.5416666666666667 * history[1][j] -
                                 2.4583333333333335 * history[2][j] +
                                 2.2916666666666665 * history[3][j])
    return evolved_vars


def adams_bashforth(order, history, evolved_vars, dvars_dt, dt):
    """
    A simple 1st, 2nd, 3rd, or 4th order explicit Adams-Bashforth
    time stepper.
    """
    history.append(dvars_dt)

    if len(history) > order:
        history.pop(0)

    if len(history) == 1:
        return _adams_bashforth1(history, evolved_vars, dvars_dt, dt)
    elif len(history) == 2:
        return _adams_bashforth2(history, evolved_vars, dvars_dt, dt)
    elif len(history) == 3:
        return _adams_bashforth3(history, evolved_vars, dvars_dt, dt)
    elif len(history) == 4:
        return _adams_bashforth4(history, evolved_vars, dvars_dt, dt)
    else:
        raise ValueError("Order must be 1, 2, 3, or 4, not %d" % order)
