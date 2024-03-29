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
class Symmetry(enum.Enum):
    No = enum.auto()
    Cylindrical = enum.auto()
    Spherical = enum.auto()


@enum.unique
class NumericalFlux(enum.IntEnum):
    Rusanov = enum.auto()
    Hll = enum.auto()
    Hllc = enum.auto()
    Hlle = enum.auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return NumericalFlux[s]
        except KeyError:
            return s


@enum.unique
class InitialData(enum.IntEnum):
    InteractingBlastWaves = enum.auto()
    Sod = enum.auto()
    LaneEmden = enum.auto()
    Lax = enum.auto()
    LeBlanc = enum.auto()
    Mach1200 = enum.auto()
    Problem123 = enum.auto()
    Sedov = enum.auto()
    Severe1 = enum.auto()
    Severe2 = enum.auto()
    Severe3 = enum.auto()
    Severe4 = enum.auto()
    Severe5 = enum.auto()
    ShuOsher = enum.auto()
    ShuOsher2 = enum.auto()
    Sinusoid = enum.auto()
    Strong = enum.auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return InitialData[s]
        except KeyError:
            return s


@enum.unique
class BoundaryCondition(enum.IntEnum):
    Constant = enum.auto()
    Periodic = enum.auto()
    Reflecting = enum.auto()


_gamma = None
_numerical_flux = None
_symmetry_alpha = None


def set_gamma(gamma):
    global _gamma
    _gamma = gamma


def get_gamma():
    return _gamma


def set_numerical_flux(numerical_flux):
    global _numerical_flux
    _numerical_flux = numerical_flux


def set_symmetry(symmetry):
    global _symmetry_alpha
    if symmetry == Symmetry.No:
        _symmetry_alpha = int(0)
    elif symmetry == Symmetry.Cylindrical:
        _symmetry_alpha = int(1)
    elif symmetry == Symmetry.Spherical:
        _symmetry_alpha = int(2)
    else:
        raise ValueError("Unknown symmetry option %s" % symmetry)


def get_symmetry():
    return _symmetry_alpha


def riemann_left_right_states(initial_data):
    """
    Returns:
    (left_mass_density, left_velocity, left_pressure,
     right_mass_density, right_velocity, right_pressure)
    """
    if initial_data == InitialData.Sod:
        return (1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    elif initial_data == InitialData.Lax:
        return (0.445, 0.698, 3.528, 0.5, 0.0, 0.571)
    elif initial_data == InitialData.Strong:
        # https://iopscience.iop.org/article/10.1086/317361
        return (10.0, 0.0, 100.0, 1.0, 0.0, 1.0)
    elif initial_data == InitialData.Problem123:
        return (1.0, -2.0, 0.1, 1.0, 2.0, 0.1)
    elif (initial_data == InitialData.Severe1
          or initial_data == InitialData.Severe2
          or initial_data == InitialData.Severe3
          or initial_data == InitialData.Severe4
          or initial_data == InitialData.Severe5):
        left_mass = 1.0
        left_velocity = 0.0
        right_mass = 1.0
        right_velocity = 0.0
        right_pressure = 0.1
        if initial_data == InitialData.Severe1:
            return (left_mass, left_velocity, 1.0e1, right_mass,
                    right_velocity, right_pressure)
        elif initial_data == InitialData.Severe2:
            return (left_mass, left_velocity, 1.0e2, right_mass,
                    right_velocity, right_pressure)
        elif initial_data == InitialData.Severe3:
            return (left_mass, left_velocity, 1.0e3, right_mass,
                    right_velocity, right_pressure)
        elif initial_data == InitialData.Severe4:
            return (left_mass, left_velocity, 1.0e4, right_mass,
                    right_velocity, right_pressure)
        elif initial_data == InitialData.Severe5:
            return (left_mass, left_velocity, 1.0e5, right_mass,
                    right_velocity, right_pressure)
    elif initial_data == InitialData.LeBlanc:
        return (1.0, 0.0, (2. / 3.) * 1.0e-1, 1.0e-3, 0.0, (2. / 3.) * 1.0e-10)
    elif initial_data == InitialData.Mach1200:
        return (1.0, 400, (2. / 3.) * 1.0e-1, 10.0, 0.0, (2. / 3.) * 1.0e-15)
    raise Exception("Non-Riemann problem initial data: " + str(initial_data))


def is_riemann_problem(problem):
    """
    Returns true if `problem` is a Riemann problem
    """
    return (problem != InitialData.ShuOsher
            and problem != InitialData.ShuOsher2
            and problem != InitialData.Sedov
            and problem != InitialData.Sinusoid
            and problem != InitialData.InteractingBlastWaves
            and problem != InitialData.LaneEmden)


def set_initial_data(num_points, initial_data):
    """
    Returns:
    (initial time, final time, x, periodic BCs (True/False), mass density,
     momentum density, energy density)
    """
    def create_grid(x_min, x_max, num_points):
        dx = (x_max - x_min) / num_points
        return x_min + (np.arange(num_points) + 0.5) * dx

    boundary_condition = None

    if initial_data == InitialData.Sinusoid:
        set_gamma(1.4)
        set_symmetry(Symmetry.No)

        x = create_grid(0.0, 8.0, num_points)

        initial_time = 0.0
        final_time = 2.0

        mass_density = 1.0 + 0.2 * np.sin(np.pi * x)
        pressure = np.zeros(x.shape) + 1.0
        velocity = np.ones(x.shape)

        boundary_condition = [
            BoundaryCondition.Periodic, BoundaryCondition.Periodic
        ]
    elif initial_data == InitialData.InteractingBlastWaves:
        set_gamma(1.4)
        set_symmetry(Symmetry.No)
        x = create_grid(0.0, 1.0, num_points)

        initial_time = 0.0
        final_time = 0.038

        # start with middle values
        mass_density = np.full(len(x), 1.0)
        pressure = np.full(len(x), 0.01)
        velocity = np.full(len(x), 0.0)

        left_mask = x < 0.1
        right_mask = x >= 0.9

        mass_density[left_mask] = 1.0
        pressure[left_mask] = 1000.0
        velocity[left_mask] = 0.0

        mass_density[right_mask] = 1.0
        pressure[right_mask] = 100.0
        velocity[right_mask] = 0.0

        boundary_condition = [
            BoundaryCondition.Reflecting, BoundaryCondition.Reflecting
        ]
    elif initial_data == InitialData.LaneEmden:
        set_gamma(2.0)
        set_symmetry(Symmetry.Spherical)
        x = create_grid(0.0, 2.0 * np.pi, num_points)

        initial_time = 0.0
        # The sound speed is ~2/3 in the star, so it takes a time of ~ 1.5 pi
        # for one sound crossing time from r=0 to r=pi.
        final_time = 1.5 * np.pi * 1000.0

        star_mask = np.abs(x) < np.pi - 1.0e-19

        mass_density = np.full(len(x), 1.0e-20)
        mass_density[star_mask] = np.abs(np.sin(x[star_mask]) / x[star_mask])

        # polytropic constant K = 1, and rho_c = 1
        pressure = mass_density**2
        velocity = np.full(len(x), 0.0)

        boundary_condition = [
            BoundaryCondition.Constant, BoundaryCondition.Constant
        ]
    elif initial_data == InitialData.ShuOsher:
        set_gamma(1.4)
        set_symmetry(Symmetry.No)
        discontinuity_location = -4.0
        x = create_grid(-5.0, 5.0, num_points)
        jump_mask = x > discontinuity_location - 1.0e-10
        final_time = 1.8

        initial_time = 0.0
        mass_density = np.full(len(x), 3.857143)
        mass_density[jump_mask] = 1.0 + 0.2 * np.sin(5.0 * x[jump_mask])

        pressure = np.full(len(x), 10.33333)
        pressure[jump_mask] = 1.0

        velocity = np.full(len(x), 2.629369)
        velocity[jump_mask] = 0.0

        boundary_condition = [
            BoundaryCondition.Constant, BoundaryCondition.Constant
        ]
    elif initial_data == InitialData.ShuOsher2:
        set_gamma(1.4)
        set_symmetry(Symmetry.No)
        discontinuity_location = -4.0
        x = create_grid(-5.0, 15.0, num_points)
        jump_mask = x > discontinuity_location - 1.0e-10
        final_time = 4.8

        initial_time = 0.0
        mass_density = np.full(len(x), 3.857143)
        mass_density[jump_mask] = 1.0 + 0.2 * np.sin(5.0 * x[jump_mask])

        pressure = np.full(len(x), 10.33333)
        pressure[jump_mask] = 1.0

        velocity = np.full(len(x), 2.629369)
        velocity[jump_mask] = 0.0

        boundary_condition = [
            BoundaryCondition.Constant, BoundaryCondition.Constant
        ]
    elif initial_data == InitialData.Sedov:
        set_gamma(1.4)
        set_symmetry(Symmetry.No)
        x = create_grid(0.0, 4.0, num_points)
        final_time = 1.0e-3

        initial_time = 0.0
        dx = 3.5 * (x[1] - x[0])
        jump_mask = (x > 2.0 - dx) & (x < 2.0 + dx)

        mass_density = np.full(len(x), 1.0)

        pressure = np.full(len(x), 1.0e-20)
        # Given the energy epsilon the pressure is given by:
        # p = \epsilon * 3.0 * (\gamma - 1.0) / ((\nu + 1) \pi dx^\nu)
        # We use epsilon=3.0e6 to match the results of:
        # "Positivity-preserving method for high-order conservative
        #  schemes solving compressible Euler equations"
        # but this seems to correspond to E=2.86e6 after evolution,
        # which is not understood.
        pressure[jump_mask] = 3.0e6 * 3.0 * (get_gamma() - 1.0) / (2.0 *
                                                                   np.pi * dx)

        velocity = np.full(len(x), 0.0)
        boundary_condition = [
            BoundaryCondition.Constant, BoundaryCondition.Constant
        ]
    else:
        set_gamma(1.4)
        set_symmetry(Symmetry.No)
        x = create_grid(0.0, 1.0, num_points)
        discontinuity_location = 0.5
        jump_mask = x > discontinuity_location - 1.0e-10

        initial_time = 0.0
        if initial_data == InitialData.Sod:
            final_time = 0.2
        elif initial_data == InitialData.Lax:
            final_time = 0.1
        elif initial_data == InitialData.Strong:
            final_time = 0.4
            # https://iopscience.iop.org/article/10.1086/317361
        elif initial_data == InitialData.Problem123:
            final_time = 0.1
        elif initial_data == InitialData.Severe1:
            final_time = 0.1
        elif initial_data == InitialData.Severe2:
            final_time = 0.03
        elif initial_data == InitialData.Severe3:
            final_time = 0.01
        elif initial_data == InitialData.Severe4:
            final_time = 0.003
        elif initial_data == InitialData.Severe5:
            final_time = 0.001
        elif initial_data == InitialData.LeBlanc:
            x = create_grid(0.0, 9.0, num_points)
            discontinuity_location = 3.0
            jump_mask = x > discontinuity_location - 1.0e-10

            set_gamma(5. / 3.)
            final_time = 6.0
        elif initial_data == InitialData.Mach1200:
            set_gamma(5. / 3.)
            final_time = 0.04

        left_mass, left_velocity, left_pressure, \
            right_mass, right_velocity, right_pressure = \
                riemann_left_right_states(initial_data)
        mass_density = np.full(len(x), left_mass)
        mass_density[jump_mask] = right_mass

        pressure = np.full(len(x), left_pressure)
        pressure[jump_mask] = right_pressure

        velocity = np.full(len(x), left_velocity)
        velocity[jump_mask] = right_velocity

        boundary_condition = [
            BoundaryCondition.Constant, BoundaryCondition.Constant
        ]

    mass_density, momentum_density, energy_density = compute_conserved(
        np.asarray([mass_density, velocity, pressure]))
    return (initial_time, final_time, x, boundary_condition, mass_density,
            momentum_density, energy_density)


def compute_pressure(mass_density, momentum_density, energy_density):
    return (_gamma - 1.0) * (energy_density -
                             0.5 * momentum_density**2 / mass_density)


def compute_energy_density(mass_density, momentum_density, pressure):
    return pressure / (_gamma - 1.0) + 0.5 * momentum_density**2 / mass_density


def compute_sound_speed(mass_density, pressure):
    return np.sqrt(_gamma * pressure / mass_density)


def compute_flux(mass_density, momentum_density, energy_density, pressure):
    mass_density_flux = momentum_density
    momentum_density_flux = momentum_density**2 / mass_density + pressure
    energy_density_flux = (energy_density +
                           pressure) * momentum_density / mass_density
    return np.asarray(
        [mass_density_flux, momentum_density_flux, energy_density_flux])


def _rusanov_helper(lf_speed, mass_f, momentum_f, energy_f,
                    reconstructed_mass_density, reconstructed_momentum_density,
                    reconstructed_energy_density):
    nf_mass = np.zeros(len(reconstructed_mass_density) // 2)
    nf_momentum = np.zeros(len(reconstructed_mass_density) // 2)
    nf_energy = np.zeros(len(reconstructed_mass_density) // 2)

    for i in range(0, len(nf_mass), 1):
        speed = max(lf_speed[2 * i + 1], lf_speed[2 * i])
        nf_mass[i] = 0.5 * (mass_f[2 * i + 1] + mass_f[
            2 * i]) - 0.5 * speed * (reconstructed_mass_density[2 * i + 1] -
                                     reconstructed_mass_density[2 * i])
        nf_momentum[i] = 0.5 * (momentum_f[2 * i + 1] +
                                momentum_f[2 * i]) - 0.5 * speed * (
                                    reconstructed_momentum_density[2 * i + 1] -
                                    reconstructed_momentum_density[2 * i])
        nf_energy[i] = 0.5 * (energy_f[2 * i + 1] + energy_f[
            2 * i]) - 0.5 * speed * (reconstructed_energy_density[2 * i + 1] -
                                     reconstructed_energy_density[2 * i])

    return (nf_mass, nf_momentum, nf_energy)


def _hll_helper(v_p_cs, v_m_cs, mass_f, momentum_f, energy_f,
                reconstructed_mass_density, reconstructed_momentum_density,
                reconstructed_energy_density):
    nf_mass = np.zeros(len(reconstructed_mass_density) // 2)
    nf_momentum = np.zeros(len(reconstructed_mass_density) // 2)
    nf_energy = np.zeros(len(reconstructed_mass_density) // 2)

    for i in range(0, len(nf_mass), 1):
        max_speed = max(max(v_p_cs[2 * i + 1], v_p_cs[2 * i]), 0.0)
        min_speed = min(min(v_m_cs[2 * i + 1], v_m_cs[2 * i]), 0.0)
        speed_diff = max_speed - min_speed
        if speed_diff == 0.0:
            continue
        nf_mass[i] = (-min_speed * mass_f[2 * i + 1] + max_speed *
                      mass_f[2 * i]) / speed_diff + max_speed * min_speed * (
                          reconstructed_mass_density[2 * i + 1] -
                          reconstructed_mass_density[2 * i]) / speed_diff
        nf_momentum[i] = (
            -min_speed * momentum_f[2 * i + 1] + max_speed *
            momentum_f[2 * i]) / speed_diff + max_speed * min_speed * (
                reconstructed_momentum_density[2 * i + 1] -
                reconstructed_momentum_density[2 * i]) / speed_diff
        nf_energy[i] = (-min_speed * energy_f[2 * i + 1] +
                        max_speed * energy_f[2 * i]
                        ) / speed_diff + max_speed * min_speed * (
                            reconstructed_energy_density[2 * i + 1] -
                            reconstructed_energy_density[2 * i]) / speed_diff

    return (nf_mass, nf_momentum, nf_energy)


def _hlle_helper(velocity, sound_speed, mass_f, momentum_f, energy_f,
                 reconstructed_mass_density, reconstructed_momentum_density,
                 reconstructed_energy_density):
    nf_mass = np.zeros(len(reconstructed_mass_density) // 2)
    nf_momentum = np.zeros(len(reconstructed_mass_density) // 2)
    nf_energy = np.zeros(len(reconstructed_mass_density) // 2)

    for i in range(0, len(nf_mass), 1):
        sqrt_rho_l = np.sqrt(reconstructed_mass_density[2 * i])
        sqrt_rho_r = np.sqrt(reconstructed_mass_density[2 * i + 1])
        inv_sqrt_rho_sum = 1.0 / (sqrt_rho_l + sqrt_rho_r)
        velocity_roe = (sqrt_rho_l * velocity[2 * i] +
                        sqrt_rho_r * velocity[2 * i + 1]) * inv_sqrt_rho_sum
        d_bar = np.sqrt((sqrt_rho_l * sound_speed[2 * i]**2 + sqrt_rho_r *
                         sound_speed[2 * i + 1]**2) * inv_sqrt_rho_sum +
                        0.5 * sqrt_rho_l * sqrt_rho_r * inv_sqrt_rho_sum**2 *
                        (velocity[2 * i + 1] - velocity[2 * i])**2)

        max_speed = max(velocity_roe + d_bar, 0.0)
        min_speed = min(velocity_roe - d_bar, 0.0)
        speed_diff = max_speed - min_speed
        nf_mass[i] = (-min_speed * mass_f[2 * i + 1] + max_speed *
                      mass_f[2 * i]) / speed_diff + max_speed * min_speed * (
                          reconstructed_mass_density[2 * i + 1] -
                          reconstructed_mass_density[2 * i]) / speed_diff
        nf_momentum[i] = (
            -min_speed * momentum_f[2 * i + 1] + max_speed *
            momentum_f[2 * i]) / speed_diff + max_speed * min_speed * (
                reconstructed_momentum_density[2 * i + 1] -
                reconstructed_momentum_density[2 * i]) / speed_diff
        nf_energy[i] = (-min_speed * energy_f[2 * i + 1] +
                        max_speed * energy_f[2 * i]
                        ) / speed_diff + max_speed * min_speed * (
                            reconstructed_energy_density[2 * i + 1] -
                            reconstructed_energy_density[2 * i]) / speed_diff

    return (nf_mass, nf_momentum, nf_energy)


def _hllc_helper(velocity, sound_speed, pressure, mass_f, momentum_f, energy_f,
                 reconstructed_mass_density, reconstructed_momentum_density,
                 reconstructed_energy_density):
    nf_mass = np.zeros(len(reconstructed_mass_density) // 2)
    nf_momentum = np.zeros(len(reconstructed_mass_density) // 2)
    nf_energy = np.zeros(len(reconstructed_mass_density) // 2)

    for i in range(0, len(nf_mass), 1):
        rho_l = reconstructed_mass_density[2 * i]
        rho_r = reconstructed_mass_density[2 * i + 1]
        u_l = velocity[2 * i]
        u_r = velocity[2 * i + 1]

        # "Simple" wave speeds which are most dissipative but also most robust,
        # especially for near-vacuum problems like the Sedov and 1-2-3 problem.
        s_r = max(velocity[2 * i + 1] + sound_speed[2 * i + 1],
                  velocity[2 * i] + sound_speed[2 * i])
        s_l = min(velocity[2 * i + 1] - sound_speed[2 * i + 1],
                  velocity[2 * i] - sound_speed[2 * i])

        # # First Einfeldt method: wave speeds based on Roe averages
        # sqrt_rho_l = np.sqrt(rho_l)
        # sqrt_rho_r = np.sqrt(rho_r)
        # inv_sqrt_rho_sum = 1.0 / (sqrt_rho_l + sqrt_rho_r)
        # velocity_roe = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) * inv_sqrt_rho_sum
        # pressure_l = (_gamma - 1.0) * (reconstructed_energy_density[2 * i] -
        #                                0.5 * u_l**2 * rho_l)
        # pressure_r = (_gamma - 1.0) * (
        #     reconstructed_energy_density[2 * i + 1] - 0.5 * u_r**2 * rho_r)

        # h_l = (reconstructed_energy_density[2 * i] + pressure_l) / rho_l
        # h_r = (reconstructed_energy_density[2 * i + 1] + pressure_r) / rho_r
        # h_tilde = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) * inv_sqrt_rho_sum

        # sound_speed_tilde = (_gamma - 1.0) * (h_tilde - 0.5 * velocity_roe**2)

        # s_r = max(velocity[2 * i + 1] + sound_speed[2 * i + 1],
        #           velocity_roe + sound_speed_tilde)
        # s_l = min(velocity_roe - sound_speed_tilde,
        #           velocity[2 * i] - sound_speed[2 * i])

        # # The second Einfeldt method for estimating the speeds.
        # sqrt_rho_l = np.sqrt(rho_l)
        # sqrt_rho_r = np.sqrt(rho_r)
        # inv_sqrt_rho_sum = 1.0 / (sqrt_rho_l + sqrt_rho_r)
        # velocity_roe = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) * inv_sqrt_rho_sum
        # d_bar = np.sqrt((sqrt_rho_l * sound_speed[2 * i]**2 + sqrt_rho_r *
        #                  sound_speed[2 * i + 1]**2) * inv_sqrt_rho_sum +
        #                 0.5 * sqrt_rho_l * sqrt_rho_r * inv_sqrt_rho_sum**2 *
        #                 (u_r - u_l)**2)
        # s_r = velocity_roe + d_bar
        # s_l = velocity_roe - d_bar

        # s_r = max(
        #     max(velocity[2 * i + 1] + sound_speed[2 * i + 1],
        #         velocity[2 * i] + sound_speed[2 * i]), 0.0)
        # s_l = min(
        #     min(velocity[2 * i + 1] - sound_speed[2 * i + 1],
        #         velocity[2 * i] - sound_speed[2 * i]), 0.0)

        if s_l >= 0.0:
            nf_mass[i] = mass_f[2 * i]
            nf_momentum[i] = momentum_f[2 * i]
            nf_energy[i] = energy_f[2 * i]
        elif s_r <= 0.0:
            nf_mass[i] = mass_f[2 * i + 1]
            nf_momentum[i] = momentum_f[2 * i + 1]
            nf_energy[i] = energy_f[2 * i + 1]
        else:
            s_star = (pressure[2 * i + 1] - pressure[2 * i] + rho_l * u_l *
                      (s_l - u_l) - rho_r * u_r *
                      (s_r - u_r)) / (rho_l * (s_l - u_l) - rho_r *
                                      (s_r - u_r))

            if s_star >= 0.0:
                # left star state
                factor_a = (s_l - velocity[2 * i]) / (s_l - s_star)

                nf_mass[i] = mass_f[2 * i] + s_l * (factor_a - 1.0) * rho_l
                nf_momentum[i] = momentum_f[
                    2 * i] + s_l * (s_star * factor_a * rho_l -
                                    reconstructed_momentum_density[2 * i])
                nf_energy[i] = energy_f[2 * i] + s_l * (
                    (reconstructed_energy_density[2 * i] +
                     (s_star - velocity[2 * i]) *
                     (rho_l * s_star + pressure[2 * i] /
                      (s_l - velocity[2 * i]))) * factor_a -
                    reconstructed_energy_density[2 * i])
            else:
                # right star state
                factor_a = (s_r - velocity[2 * i + 1]) / (s_r - s_star)

                nf_mass[i] = mass_f[2 * i + 1] + s_r * (factor_a - 1.0) * rho_r
                nf_momentum[i] = momentum_f[2 * i + 1] + s_r * (
                    s_star * factor_a * rho_r -
                    reconstructed_momentum_density[2 * i + 1])
                nf_energy[i] = energy_f[2 * i + 1] + s_r * (
                    (reconstructed_energy_density[2 * i + 1] +
                     (s_star - velocity[2 * i + 1]) *
                     (rho_r * s_star + pressure[2 * i + 1] /
                      (s_r - velocity[2 * i + 1]))) * factor_a -
                    reconstructed_energy_density[2 * i + 1])

    return (nf_mass, nf_momentum, nf_energy)


# If numba is present, JIT the nuumerical flux helper
if use_numba:
    rusanov_helper = nb.jit(nopython=True)(_rusanov_helper)
    hll_helper = nb.jit(nopython=True)(_hll_helper)
    hlle_helper = nb.jit(nopython=True)(_hlle_helper)
    hllc_helper = nb.jit(nopython=True)(_hllc_helper)
else:
    rusanov_helper = _rusanov_helper
    hll_helper = _hll_helper
    hlle_helper = _hlle_helper
    hllc_helper = _hllc_helper


def compute_numerical_flux(mass_density, momentum_density, energy_density,
                           mass_f, momentum_f, energy_f, velocity, pressure,
                           sound_speed):
    if _numerical_flux == NumericalFlux.Rusanov:
        lf_speed = abs(velocity) + sound_speed

        return rusanov_helper(lf_speed, mass_f, momentum_f, energy_f,
                              mass_density, momentum_density, energy_density)
    elif _numerical_flux == NumericalFlux.Hll:
        velocity_p_cs = velocity + sound_speed
        velocity_m_cs = velocity - sound_speed

        return hll_helper(velocity_p_cs, velocity_m_cs, mass_f, momentum_f,
                          energy_f, mass_density, momentum_density,
                          energy_density)
    elif _numerical_flux == NumericalFlux.Hlle:
        return hlle_helper(velocity, sound_speed, mass_f, momentum_f, energy_f,
                           mass_density, momentum_density, energy_density)
    elif _numerical_flux == NumericalFlux.Hllc:
        return hllc_helper(velocity, sound_speed, pressure, mass_f, momentum_f,
                           energy_f, mass_density, momentum_density,
                           energy_density)
    else:
        raise ValueError("Unknown numerical flux")


def compute_sources(radius, evolved_vars):
    """
    Computes source terms for the symmetry.
    """
    mass_density = evolved_vars[0]
    momentum_density = evolved_vars[1]
    energy_density = evolved_vars[2]
    factor = -_symmetry_alpha / radius
    pressure = compute_pressure(mass_density, momentum_density, energy_density)
    return (factor * momentum_density,
            factor * momentum_density**2 / mass_density, factor *
            (energy_density + pressure) * momentum_density / mass_density)


def compute_primitives(primitive_vars, evolved_vars):
    """
    primitive_vars[0] = rest mass density
    primitive_vars[1] = velocity in x-direction
    primitive_vars[2] = pressure

    evolved_vars[0] = rest mass density
    evolved_vars[1] = momentum density in x-direction
    evolved_vars[2] = energy density
    """
    if primitive_vars is None or not np.array_equal(primitive_vars.shape,
                                                    evolved_vars.shape):
        primitive_vars = np.zeros(evolved_vars.shape)

    primitive_vars[0] = evolved_vars[0]
    primitive_vars[1] = evolved_vars[1] / evolved_vars[0]
    primitive_vars[2] = (_gamma - 1.0) * (
        evolved_vars[2] - 0.5 * evolved_vars[1]**2 / evolved_vars[0])
    return primitive_vars


def compute_conserved(primitive_vars):
    """
    primitive_vars[0] = rest mass density
    primitive_vars[1] = velocity in x-direction
    primitive_vars[2] = pressure

    evolved_vars[0] = rest mass density
    evolved_vars[1] = momentum density in x-direction
    evolved_vars[2] = energy density
    """
    return np.asarray([
        primitive_vars[0], primitive_vars[1] * primitive_vars[0],
        compute_energy_density(primitive_vars[0],
                               primitive_vars[1] * primitive_vars[0],
                               primitive_vars[2])
    ])
