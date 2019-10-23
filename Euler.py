# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import time as time_mod
import numpy as np

use_numba = True
try:
    import numba as nb
except:
    use_numba = False

import Reconstruction as recons
import TimeStepper
import Derivative
import NewtonianEuler as ne
import SedovSolution as sedov
import Plotting as plot

################################################################
# Configuration:
reconstruct_prims = True
# End configuration
################################################################


class NewtonianEuler1d:
    _reconstruct_prims = None

    _x = None
    _dx = None

    _periodic_bcs = None
    _buffer_size = 9

    _reconstructor = None
    _reconstruction_scheme = None
    _deriv_scheme = None

    _order_used = None

    def __init__(self, reconstruct_prims, x, periodic_bcs, reconstructor,
                 reconstruction_scheme, deriv_scheme):
        self._reconstruct_prims = reconstruct_prims

        self._x = np.copy(x)
        self._dx = x[1] - x[0]

        self._periodic_bcs = periodic_bcs

        self._reconstructor = reconstructor
        self._reconstruction_scheme = reconstruction_scheme
        self._deriv_scheme = deriv_scheme

        self._order_used = np.zeros(len(x), dtype=int)

    def _reset_order_used(self, length=None):
        if length is None:
            length = len(self._x)

        # set to 100 because during reconstruction we use the min of the
        # current order used and the order used by past variables. It seems
        # unlikely we will use a 100th order scheme.
        if len(self._order_used) != length:
            self._order_used = np.zeros(length, dtype=int) + 100
        else:
            self._order_used[:] = 100

    def get_dx(self):
        return self._dx

    def _extend_for_periodic(self, variables):
        new_vars = np.zeros(
            [variables.shape[0], variables.shape[1] + 2 * self._buffer_size])
        for i in range(variables.shape[0]):
            # copy over center
            new_vars[i, self._buffer_size:-self._buffer_size] = variables[i, :]
            # copy over boundaries
            new_vars[i, 0:self.
                     _buffer_size] = variables[i, -self._buffer_size:]
            new_vars[i, -self._buffer_size:] = variables[i, 0:self.
                                                         _buffer_size]
        return new_vars

    def __call__(self, evolved_vars, time):
        self._reset_order_used(
            len(self._x) +
            2 * self._buffer_size if self._periodic_bcs else len(self._x))
        if self._reconstruct_prims:
            primitive_vars = None
            primitive_vars = ne.compute_primitives(primitive_vars,
                                                   evolved_vars)
            if self._periodic_bcs:
                primitive_vars = self._extend_for_periodic(primitive_vars)

            recons_prim = self._reconstructor(primitive_vars,
                                              self._reconstruction_scheme,
                                              self._order_used)
            recons_conserved = ne.compute_conserved(recons_prim)
        else:
            recons_conserved = self._reconstructor(evolved_vars,
                                                   self._reconstruction_scheme,
                                                   self._order_used)

        numerical_fluxes_at_faces = ne.compute_numerical_flux(recons_conserved)

        dt_evolved_vars = -1.0 * self._flux_deriv(
            numerical_fluxes_at_faces,
            ne.compute_flux(
                self._extend_for_periodic(evolved_vars) if self.
                _periodic_bcs else evolved_vars))

        if ne.get_symmetry() != 0:
            dt_evolved_vars += ne.compute_sources(self._x, evolved_vars)

        if self._periodic_bcs:
            new_dt_evolved_vars = np.zeros(evolved_vars.shape)
            for i in range(dt_evolved_vars.shape[0]):
                # copy over center
                new_dt_evolved_vars[i, :] = dt_evolved_vars[
                    i, self._buffer_size:-self._buffer_size]
            dt_evolved_vars = new_dt_evolved_vars
        else:
            bc_distance = self._buffer_size
            # zero time derivs at boundary
            for i in range(len(dt_evolved_vars)):
                dt_evolved_vars[i][0:bc_distance] = 0.0
                dt_evolved_vars[i][-bc_distance:] = 0.0
        return dt_evolved_vars

    def get_order_used(self):
        if self._periodic_bcs:
            return self._order_used[self._buffer_size:-self._buffer_size]
        return self._order_used

    def _flux_deriv(self, numerical_fluxes_at_faces, cell_fluxes):
        return Derivative.differentiate_flux(self._deriv_scheme, self._dx,
                                             numerical_fluxes_at_faces,
                                             cell_fluxes, self._order_used)


def do_solve(num_cells, problem, cfl, reconstructor, reconstruction_scheme,
             deriv_scheme):
    time, final_time, x, periodic_bcs, mass_density, \
        momentum_density, energy_density = ne.set_initial_data(
            num_cells, problem)

    ne1d_solver = NewtonianEuler1d(reconstruct_prims, x, periodic_bcs,
                                   reconstructor, reconstruction_scheme,
                                   deriv_scheme)

    stepper = TimeStepper.Rk3Ssp(
        ne1d_solver,
        np.asarray([mass_density, momentum_density, energy_density]), time)

    while stepper.get_time() <= final_time:
        mass_density, momentum_density, energy_density = stepper.get_evolved_vars(
        )
        sound_speed = ne.compute_sound_speed(
            mass_density,
            ne.compute_pressure(mass_density, momentum_density,
                                energy_density))
        speed = np.amax(np.abs(momentum_density / mass_density) + sound_speed)
        dt = cfl * stepper.get_cfl_coefficient() * ne1d_solver.get_dx() / speed
        if stepper.get_time() + dt > final_time:
            dt = final_time - stepper.get_time()
            stepper.take_step(dt)
            break
        stepper.take_step(dt)

    mass_density, momentum_density, energy_density = stepper.get_evolved_vars()
    return (stepper.get_time(), x, ne1d_solver.get_order_used(), mass_density,
            momentum_density, energy_density)


def main(problem, num_cells, numerical_flux, cfl):
    ne.set_numerical_flux(numerical_flux)
    print("Starting solves...")
    start = time_mod.time()
    time, x, order_used, mass_density, momentum_density, energy_density \
        = do_solve(num_cells, problem, cfl, recons.reconstruct, recons.Scheme.Wcns3,
                   Derivative.Scheme.MD)
    print("Time for ND5: ", time_mod.time() - start)

    # Set global order at the boundaries to 9 to avoid weird plots
    order_used[order_used > 10] = 9
    every_n = 1
    if ne.is_riemann_problem(problem):
        import NewtonianRiemannSolver as riemann
        discontinuity_location = 3.0 if problem == ne.InitialData.LeBlanc else None
        mass_density_ref, velocity_ref, pressure_ref = \
            riemann.riemann_problem_solution(
                *ne.riemann_left_right_states(problem),
                x, time, ne.get_gamma(),
                discontinuity_location=discontinuity_location)
        x_ref = x
        exact_or_ref_plot_label = "Exact"
    elif problem == ne.InitialData.ShuOsher:
        # Compute reference solution for Shu osher
        num_cells_original = num_cells
        num_cells = 5000
        _, x_ref, _, mass_density_ref, momentum_density_ref, \
            energy_density_ref = do_solve(num_cells, problem, cfl,
                                          recons.reconstruct,
                                          recons.Scheme.Wcns3,
                                          Derivative.Scheme.MD)
        num_cells = num_cells_original
        velocity_ref = momentum_density_ref / mass_density_ref
        pressure_ref = ne.compute_pressure(mass_density_ref,
                                           momentum_density_ref,
                                           energy_density_ref)
        exact_or_ref_plot_label = "Reference"
    elif problem == ne.InitialData.Sedov:
        plt.plot(
            x,
            ne.compute_pressure(mass_density, momentum_density,
                                energy_density), 'o')
        r, p, rho, u, _, _, _, _, _ = sedov.sedov(t=1.0e-3,
                                                  E0=2.86e6,
                                                  rho0=1.0,
                                                  gamma=ne.get_gamma(),
                                                  nu=1)

        # Compute coordinate along entire line
        num_zero_points = 100
        center = 0.5 * x[-1]
        pressure_zero = 1.0e-20
        r = r + center
        r = np.concatenate((r, np.linspace(r[-1], 2.0 * center,
                                           num_zero_points)))
        x_ref = np.concatenate((-r[::-1] + 2.0 * center, r))

        # Compute pressure, density, and velocity along entire line
        mass_density_ref = np.concatenate(
            (rho, np.zeros([num_zero_points]) + 1.0))
        mass_density_ref = np.concatenate(
            (mass_density_ref[::-1], mass_density_ref))

        pressure_ref = np.concatenate(
            (p, np.zeros([num_zero_points]) + pressure_zero))
        pressure_ref = np.concatenate((pressure_ref[::-1], pressure_ref))

        velocity_ref = np.concatenate((u, np.zeros([num_zero_points])))
        velocity_ref = np.concatenate((-velocity_ref[::-1], velocity_ref))
        exact_or_ref_plot_label = "Exact"
    elif problem == ne.InitialData.Sinusoid:
        x_ref = x
        mass_density_ref = 1.0 + 0.2 * np.sin(np.pi * (x - time))
        velocity_ref = np.ones(len(x))
        pressure_ref = np.ones(len(x))
        exact_or_ref_plot_label = "Exact"

    plot.generate_plot_with_reference(
        x,
        x_ref,
        mass_density,
        mass_density_ref,
        "Density",
        str(problem).replace("InitialData.", '') + "Density" + str(num_cells) +
        ".pdf",
        exact_or_ref_plot_label,
        every_n,
        set_log_y=(problem == ne.InitialData.LeBlanc))
    plot.generate_plot_with_reference(
        x,
        x_ref,
        ne.compute_pressure(mass_density, momentum_density, energy_density),
        pressure_ref,
        "Pressure",
        str(problem).replace("InitialData.", '') + "Pressure" +
        str(num_cells) + ".pdf",
        exact_or_ref_plot_label,
        every_n,
        set_log_y=(problem == ne.InitialData.LeBlanc))
    plot.generate_plot_with_reference(
        x, x_ref, momentum_density / mass_density, velocity_ref, "Velocity",
        str(problem).replace("InitialData.", '') + "Velocity" +
        str(num_cells) + ".pdf", exact_or_ref_plot_label, every_n)
    plot.generate_plot_with_reference(
        x, x_ref, order_used, None, "Local Order",
        str(problem).replace("InitialData.", '') + "Order" + str(num_cells) +
        ".pdf", exact_or_ref_plot_label, every_n)


def parse_args():
    """
    Parse arguments and return a dictionary of them.
    """
    import argparse as ap
    parser = ap.ArgumentParser(
        "Evolve one of the standard compressible Newtonian Euler problems.\n"
        "\n"
        "Output PDF plots are written with the analytic/reference solution\n"
        "overlaid on top of the numerical solution. The naming convention\n"
        "for the files is \"ProblemNameQuantityResultion.pdf\".\n\n"
        "A good CFL starting value is 0.8. We recommend using the HLL\n"
        "approximate Riemann solver. Some reasonable resolutions for\n"
        "different problems are:\n"
        "  Sod: 200\n"
        "  Lax: 200\n"
        "  ShuOsher: 400\n"
        "  Severe*: 200\n"
        "  1-2-3 problem: 400\n"
        "  LeBlanc: 800\n"
        "  Sedov: 800\n")
    parser.add_argument('--problem',
                        type=ne.InitialData.argparse,
                        choices=list(ne.InitialData),
                        required=True,
                        help='Which standard test problem to solve.')
    parser.add_argument(
        '--numerical-flux',
        required=False,
        type=ne.NumericalFlux.argparse,
        default=ne.NumericalFlux.Hll,
        choices=list(ne.NumericalFlux),
        help='The numerical flux to use to solve the (approximate) '
        'Riemann problem.')
    parser.add_argument('-n',
                        '--number-of-cells',
                        type=int,
                        required=True,
                        help='The number of grid points/cells to use on the '
                        'finite-difference grid.')
    parser.add_argument('--cfl',
                        type=float,
                        required=True,
                        help='The CFL factor to use.')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    main(args['problem'], args['number_of_cells'], args['numerical_flux'],
         args['cfl'])
