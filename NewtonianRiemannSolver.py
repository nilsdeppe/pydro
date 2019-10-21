# solve the Riemann problem for a gamma-law gas

from __future__ import print_function

import enum
import numpy as np
import scipy.optimize as optimize


@enum.unique
class _Side(enum.Enum):
    Right = enum.auto()
    Left = enum.auto()


class _State:
    side = None
    density = None
    pressure = None
    velocity = None

    def __init__(self, side, density, pressure, velocity):
        self.side = side
        self.density = density
        self.pressure = pressure
        self.velocity = velocity


def _hugoniot_velocity(pressure, gamma, state):
    """
    Compute the velocity/Hugoniot curve as a function of the pressure.
    """

    if state.side == _Side.Left:
        side_sign = 1.0
    elif state.side == _Side.Right:
        side_sign = -1.0

    sound_speed = np.sqrt(gamma * state.pressure / state.density)

    if pressure < state.pressure:
        # Rarefaction wave
        velocity = (state.velocity + side_sign * (2.0 * sound_speed /
                                                  (gamma - 1.0)) *
                    (1.0 - (pressure / state.pressure)**((gamma - 1.0) /
                                                         (2.0 * gamma))))
    else:
        # Shock wave
        beta = (gamma + 1.0) / (gamma - 1.0)
        velocity = (state.velocity + side_sign *
                    (2.0 * sound_speed / np.sqrt(2.0 * gamma *
                                                 (gamma - 1.0))) *
                    (1.0 - pressure / state.pressure) /
                    np.sqrt(1.0 + beta * pressure / state.pressure))

    return velocity


def _find_star_state(gamma, pressure_min, pressure_max, left_state,
                     right_state):
    """
    Find the star pressure using a root find on the Hugoniot curve,
    then compute the star velocity

    pressure_min and pressure_max must enclose the root of the Hugoniot curve.
    """
    def temp_func(pressure):
        return (_hugoniot_velocity(pressure, gamma, left_state) -
                _hugoniot_velocity(pressure, gamma, right_state))

    if temp_func(pressure_min) * temp_func(pressure_max) >= 0.0:
        # If there is no sign change then the pressure is zero
        pressure_star = 0.0
    else:
        # Find the star pressure by a root find
        pressure_star = optimize.brentq(temp_func, pressure_min, pressure_max)

    return (pressure_star, _hugoniot_velocity(pressure_star, gamma,
                                              left_state))


def riemann_problem_solution(left_density,
                             left_velocity,
                             left_pressure,
                             right_density,
                             right_velocity,
                             right_pressure,
                             x,
                             time,
                             gamma,
                             discontinuity_location=None,
                             p_min=0.0,
                             p_max=1.0e30):
    """
    Solves a 1d Riemann problem with given left and right states.
    x is the grid, gamma the adiabatic index of the ideal gas,
    discontinuity_location is taken to be the middle of the
    domain if not specified.

    Returns: (mass density, velocity, pressure)
    """
    if discontinuity_location is None:
        discontinuity_location = 0.5 * (x[-1] - x[0])

    left_state = _State(_Side.Left, left_density, left_pressure, left_velocity)
    right_state = _State(_Side.Right, right_density, right_pressure,
                         right_velocity)

    pressure_star, velocity_star = _find_star_state(gamma, p_min, p_max,
                                                    left_state, right_state)

    # Now that we have pressure_star and velocity_star, we can compute the
    # solution over the grid.
    xi = (x - discontinuity_location) / time

    gamma_fac = (gamma - 1.0) / (gamma + 1.0)

    density = np.zeros([len(x)])
    velocity = np.zeros([len(x)])
    pressure = np.zeros([len(x)])

    for i in range(len(x)):
        if xi[i] > velocity_star:
            # We are in the R* or R region
            state = right_state
            side_sign = 1.0
        else:
            # We are in the L* or L region
            state = left_state
            side_sign = -1.0

        pressure_ratio = pressure_star / state.pressure

        sound_speed = np.sqrt(gamma * state.pressure / state.density)

        # Use 4.54 or 4.61 of Toro 2009
        sound_speed_star = sound_speed * pressure_ratio**((gamma - 1.0) /
                                                          (2.0 * gamma))

        if pressure_star > state.pressure:
            # We are in the shock region
            # Use 4.50 or 4.57 of Toro 2009
            density_star = state.density * (pressure_ratio + gamma_fac) / (
                gamma_fac * pressure_ratio + 1.0)

            # Use 4.52 or 4.59 of Toro 2009
            S = state.velocity + side_sign * sound_speed * np.sqrt(
                0.5 * (gamma + 1.0) / gamma * pressure_ratio + 0.5 *
                (gamma - 1.0) / gamma)

            if (state.side == _Side.Right
                    and xi[i] > S) or (state.side == _Side.Left and xi[i] < S):
                # We are in the region to the left or right of the shock
                density[i] = state.density
                velocity[i] = state.velocity
                pressure[i] = state.pressure
            else:
                # We are in the star region
                density[i] = density_star
                velocity[i] = velocity_star
                pressure[i] = pressure_star

        else:
            # The rarefaction wave has three parts: the head, tail and the fan.
            # We need to check which part the current grid point is in and
            # then solve appropriately.

            # Find the speed of the head and tail of the rarefaction fan
            rarafaction_speed_head = state.velocity + side_sign * sound_speed
            rarafaction_speed_tail = (velocity_star +
                                      side_sign * sound_speed_star)

            if (state.side == _Side.Right and xi[i] > rarafaction_speed_head
                ) or (state.side == _Side.Left
                      and xi[i] < rarafaction_speed_head):
                # We are in the region to the left or right of the rarefaction
                density[i] = state.density
                velocity[i] = state.velocity
                pressure[i] = state.pressure
            elif (state.side == _Side.Right and xi[i] < rarafaction_speed_tail
                  ) or (state.side == _Side.Left
                        and xi[i] > rarafaction_speed_tail):
                # We are in the star region. Use 4.53 and 4.60 from Toro 2009
                density[i] = state.density * pressure_ratio**(1.0 / gamma)
                velocity[i] = velocity_star
                pressure[i] = pressure_star
            else:
                # We are in the fan region, use 4.56 and 4.63 of Toro 2009
                if 2.0 / (gamma + 1.0) < side_sign * gamma_fac * (
                        state.velocity - xi[i]) / sound_speed:
                    density[i] = 0.0
                    pressure[i] = 0.0
                else:
                    density[i] = state.density * (
                        2.0 / (gamma + 1.0) - side_sign * gamma_fac *
                        (state.velocity - xi[i]) / sound_speed)**(
                            2.0 / (gamma - 1.0))
                    pressure[i] = state.pressure * (
                        2.0 / (gamma + 1.0) - side_sign * gamma_fac *
                        (state.velocity - xi[i]) / sound_speed)**(
                            2.0 * gamma / (gamma - 1.0))
                velocity[i] = 2.0 / (
                    gamma + 1.0) * (-side_sign * sound_speed + 0.5 *
                                    (gamma - 1.0) * state.velocity + xi[i])

    return density, velocity, pressure
