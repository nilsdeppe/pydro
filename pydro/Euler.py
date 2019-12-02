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
import LaneEmden1d

################################################################
# Configuration:
reconstruct_prims = True
# End configuration
################################################################


class NewtonianEuler1d:
    _reconstruct_prims = None

    _x = None
    _dx = None

    _boundary_conditions = None
    _buffer_size = 9

    _reconstructor = None
    _reconstruction_scheme = None
    _deriv_scheme = None

    _order_used = None

    _density_lower_bound = 1.0e-30

    def __init__(self, reconstruct_prims, x, boundary_conditions,
                 reconstructor, reconstruction_scheme, deriv_scheme):
        self._reconstruct_prims = reconstruct_prims

        self._x = np.copy(x)
        self._dx = x[1] - x[0]

        self._boundary_conditions = boundary_conditions

        self._reconstructor = reconstructor
        self._reconstruction_scheme = reconstruction_scheme
        self._deriv_scheme = deriv_scheme

        self._order_used = np.zeros(len(x), dtype=int)

    def _reset_order_used(self, length=None):
        if length is None:
            length = len(self._x)
            if self._boundary_conditions[0] != ne.BoundaryCondition.Constant:
                length += self._buffer_size
            if self._boundary_conditions[1] != ne.BoundaryCondition.Constant:
                length += self._buffer_size

        # set to 100 because during reconstruction we use the min of the
        # current order used and the order used by past variables. It seems
        # unlikely we will use a 100th order scheme.
        if len(self._order_used) != length:
            self._order_used = np.zeros(length, dtype=int) + 100
        else:
            self._order_used[:] = 100

    def get_dx(self):
        """
        Return the inter-grid-point spacing
        """
        return self._dx

    @staticmethod
    def compute_sound_speed(conserved_vars):
        return ne.compute_sound_speed(conserved_vars[0],
                                      ne.compute_pressure(*conserved_vars))

    def _extend_for_boundary_conditions(self, variables):
        """
        Extend the variables numpy matrix (rows is variables, columns
        are grid points) to account for periodic or reflecting boundary
        conditions. In the reflecting case the velocity is negated.
        """
        if all(bc == ne.BoundaryCondition.Constant
               for bc in self._boundary_conditions):
            return variables

        lower_buffer = (
            0 if self._boundary_conditions[0] == ne.BoundaryCondition.Constant
            else self._buffer_size)
        upper_buffer = (
            0 if self._boundary_conditions[1] == ne.BoundaryCondition.Constant
            else self._buffer_size)

        new_vars = np.zeros([
            variables.shape[0],
            variables.shape[1] + lower_buffer + upper_buffer
        ])
        # copy over center
        for i in range(variables.shape[0]):
            new_vars[i, lower_buffer:(len(new_vars) if upper_buffer ==
                                      0 else -upper_buffer)] = variables[i, :]
        # Check lower side for periodic BCs
        if self._boundary_conditions[0] == ne.BoundaryCondition.Periodic:
            for i in range(variables.shape[0]):
                new_vars[i, 0:self.
                         _buffer_size] = variables[i, -self._buffer_size:]
        # Check upper side for periodic BCs
        if self._boundary_conditions[1] == ne.BoundaryCondition.Periodic:
            for i in range(variables.shape[0]):
                new_vars[i, -self._buffer_size:] = variables[i, 0:self.
                                                             _buffer_size]

        # Check lower side for reflecting BCs
        if self._boundary_conditions[0] == ne.BoundaryCondition.Reflecting:
            temp = np.copy(variables[0, 0:self._buffer_size])
            new_vars[0, 0:self._buffer_size] = temp[::-1]

            temp = np.copy(variables[1, 0:self._buffer_size])
            new_vars[1, 0:self._buffer_size] = -temp[::-1]

            temp = np.copy(variables[2, 0:self._buffer_size])
            new_vars[2, 0:self._buffer_size] = temp[::-1]

        # Check upper side for reflecting BCs
        if self._boundary_conditions[1] == ne.BoundaryCondition.Reflecting:
            temp = np.copy(variables[0, -self._buffer_size:])
            new_vars[0, -self._buffer_size:] = temp[::-1]

            temp = np.copy(variables[1, -self._buffer_size:])
            new_vars[1, -self._buffer_size:] = -temp[::-1]

            temp = np.copy(variables[2, -self._buffer_size:])
            new_vars[2, -self._buffer_size:] = temp[::-1]

        return new_vars

    def _finalize_boundary_conditions(self, variables):
        """
        Removes the extra grid points for periodic or reflecting BCs and zeros
        the time derivative boundaries for Constant boundary conditions.
        """
        if all(bc == ne.BoundaryCondition.Constant
               for bc in self._boundary_conditions):
            for var in variables:
                var[0:self._buffer_size] = 0.0
                var[-self._buffer_size:] = 0.0
            return variables

        # Not just constant boundary conditions case...
        new_vars = np.zeros([variables.shape[0], len(self._x)])
        lower_buffer = (
            0 if self._boundary_conditions[0] == ne.BoundaryCondition.Constant
            else self._buffer_size)
        upper_buffer = (variables.shape[1] if self._boundary_conditions[1] ==
                        ne.BoundaryCondition.Constant else -self._buffer_size)
        for i in range(variables.shape[0]):
            if self._boundary_conditions[0] == ne.BoundaryCondition.Constant:
                variables[i, 0:self._buffer_size] = 0.0
            if self._boundary_conditions[1] == ne.BoundaryCondition.Constant:
                variables[i, -self._buffer_size:] = 0.0
            # copy over center
            new_vars[i, :] = variables[i, lower_buffer:upper_buffer]
        return new_vars

    def __call__(self, evolved_vars, time):
        self._reset_order_used()
        if self._reconstruct_prims:
            primitive_vars = None
            primitive_vars = ne.compute_primitives(primitive_vars,
                                                   evolved_vars)
            if any(bc != ne.BoundaryCondition.Constant
                   for bc in self._boundary_conditions):
                primitive_vars = self._extend_for_boundary_conditions(
                    primitive_vars)

            recons_prim = self._reconstructor(primitive_vars,
                                              self._reconstruction_scheme,
                                              self._order_used)
            recons_prim[0][recons_prim[0] == 0.0] = self._density_lower_bound
            recons_conserved = ne.compute_conserved(recons_prim)
        else:
            recons_conserved = self._reconstructor(evolved_vars,
                                                   self._reconstruction_scheme,
                                                   self._order_used)
            recons_conserved[0][recons_conserved[0] ==
                                0.0] = self._density_lower_bound
            recons_prim = ne.compute_primitives(recons_prim, recons_conserved)

        recons_flux = ne.compute_flux(mass_density=recons_conserved[0],
                                      momentum_density=recons_conserved[1],
                                      energy_density=recons_conserved[2],
                                      pressure=recons_prim[2])
        recons_sound_speed = ne.compute_sound_speed(recons_prim[0],
                                                    recons_prim[2])
        numerical_fluxes_at_faces = ne.compute_numerical_flux(
            mass_density=recons_prim[0],
            momentum_density=recons_conserved[1],
            energy_density=recons_conserved[2],
            mass_f=recons_flux[0],
            momentum_f=recons_flux[1],
            energy_f=recons_flux[2],
            velocity=recons_prim[1],
            pressure=recons_prim[2],
            sound_speed=recons_sound_speed)

        # Note: the cell-center flux computation could be elided completely
        # if MD instead of MND derivatives are used.
        cell_center_fluxes = None
        if (self._boundary_conditions !=
            [ne.BoundaryCondition.Constant, ne.BoundaryCondition.Constant]):
            extended_vars = self._extend_for_boundary_conditions(evolved_vars)
            cell_center_pressure = ne.compute_pressure(
                mass_density=extended_vars[0],
                momentum_density=extended_vars[1],
                energy_density=extended_vars[2])
            cell_center_fluxes = ne.compute_flux(
                mass_density=extended_vars[0],
                momentum_density=extended_vars[1],
                energy_density=extended_vars[2],
                pressure=cell_center_pressure)
        else:
            cell_center_pressure = ne.compute_pressure(
                mass_density=evolved_vars[0],
                momentum_density=evolved_vars[1],
                energy_density=evolved_vars[2])
            cell_center_fluxes = ne.compute_flux(
                mass_density=evolved_vars[0],
                momentum_density=evolved_vars[1],
                energy_density=evolved_vars[2],
                pressure=cell_center_pressure)

        dt_evolved_vars = -1.0 * self._flux_deriv(numerical_fluxes_at_faces,
                                                  cell_center_fluxes)

        if ne.get_symmetry() != 0:
            dt_evolved_vars += ne.compute_sources(self._x, evolved_vars)
        dt_evolved_vars = self._finalize_boundary_conditions(dt_evolved_vars)
        return dt_evolved_vars

    def get_order_used(self):
        """
        Returns the lowest order used for reconstruction.
        """
        lower_bound = (
            0 if self._boundary_conditions[0] == ne.BoundaryCondition.Constant
            else self._buffer_size)
        upper_bound = (
            len(self._order_used)
            if self._boundary_conditions[1] == ne.BoundaryCondition.Constant
            else len(self._order_used) - self._buffer_size)

        return self._order_used[lower_bound:upper_bound]

    def _flux_deriv(self, numerical_fluxes_at_faces, cell_fluxes):
        return Derivative.differentiate_flux(self._deriv_scheme, self._dx,
                                             numerical_fluxes_at_faces,
                                             cell_fluxes, self._order_used)


def do_solve(num_cells, problem, cfl, reconstructor, reconstruction_scheme,
             deriv_scheme, generate_spacetime_plots):
    time, final_time, x, boundary_conditions, mass_density, \
        momentum_density, energy_density = ne.set_initial_data(
            num_cells, problem)

    if problem == ne.InitialData.LaneEmden:
        ne1d_solver = LaneEmden1d.NewtonianEuler1dLaneEmden(
            reconstruct_prims, x, boundary_conditions, reconstructor,
            reconstruction_scheme, deriv_scheme)
    else:
        ne1d_solver = NewtonianEuler1d(reconstruct_prims, x,
                                       boundary_conditions, reconstructor,
                                       reconstruction_scheme, deriv_scheme)

    stepper = TimeStepper.Rk4Ssp(
        ne1d_solver,
        np.asarray([mass_density, momentum_density, energy_density]), time)

    times = []
    spacetime_history = [[]
                         for _ in range(len(stepper.get_evolved_vars()) + 1)]

    time_index = 0

    while stepper.get_time() <= final_time:
        mass_density, momentum_density, energy_density = stepper.get_evolved_vars(
        )
        sound_speed = ne1d_solver.compute_sound_speed(
            stepper.get_evolved_vars())
        speed = np.amax(np.abs(momentum_density / mass_density) + sound_speed)
        dt = cfl * stepper.get_cfl_coefficient() * ne1d_solver.get_dx() / speed
        if stepper.get_time() + dt > final_time:
            dt = final_time - stepper.get_time()
            stepper.take_step(dt)
            break
        stepper.take_step(dt)
        time_index += 1
        if time_index % 400 == 0:
            print("Time: ", stepper.get_time())
        if generate_spacetime_plots:
            for var_index in range(len(stepper.get_evolved_vars())):
                spacetime_history[var_index].append(
                    np.copy(stepper.get_evolved_vars()[var_index]))
            spacetime_history[-1].append(np.copy(ne1d_solver.get_order_used()))
            times.append(stepper.get_time() - dt)

    if generate_spacetime_plots:
        for i in range(len(spacetime_history)):
            spacetime_history[i] = np.asarray(spacetime_history[i])

    mass_density, momentum_density, energy_density = stepper.get_evolved_vars()
    return (stepper.get_time(), x, ne1d_solver.get_order_used(), mass_density,
            momentum_density, energy_density, np.asarray(times),
            spacetime_history)


def main(problem, num_cells, numerical_flux, cfl, generate_spacetime_plots):
    ne.set_numerical_flux(numerical_flux)
    print("Starting solves...")
    start = time_mod.time()
    time, x, order_used, mass_density, momentum_density, energy_density, \
        times, spacetime_history = do_solve(num_cells, problem, cfl,
                                            recons.reconstruct,
                                            recons.Scheme.Wcns3,
                                            Derivative.Scheme.MD,
                                            generate_spacetime_plots)
    print("Time for ND5: ", time_mod.time() - start)

    # Set global order at the boundaries to 9 to avoid weird plots
    order_used[order_used > 10] = 9
    every_n = 1
    if ne.is_riemann_problem(problem):
        import NewtonianRiemannSolver as riemann
        discontinuity_location = (3.0 if problem == ne.InitialData.LeBlanc else
                                  None)
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
            energy_density_ref, _, _ = do_solve(num_cells, problem, cfl,
                                                recons.reconstruct,
                                                recons.Scheme.Wcns3,
                                                Derivative.Scheme.MD,
                                                generate_spacetime_plots)
        num_cells = num_cells_original
        velocity_ref = momentum_density_ref / mass_density_ref
        pressure_ref = ne.compute_pressure(mass_density_ref,
                                           momentum_density_ref,
                                           energy_density_ref)
        exact_or_ref_plot_label = "Reference"
    elif problem == ne.InitialData.InteractingBlastWaves:
        # Compute reference solution for interacting blast waves
        print("Generating high-resolution reference solution. "
              "This will take a few minutes...")
        num_cells_original = num_cells
        num_cells = 5000
        _, x_ref, _, mass_density_ref, momentum_density_ref, \
            energy_density_ref, _, _ = do_solve(num_cells, problem, cfl,
                                                recons.reconstruct,
                                                recons.Scheme.Wcns3,
                                                Derivative.Scheme.MD,
                                                generate_spacetime_plots)
        num_cells = num_cells_original
        velocity_ref = momentum_density_ref / mass_density_ref
        pressure_ref = ne.compute_pressure(mass_density_ref,
                                           momentum_density_ref,
                                           energy_density_ref)
        exact_or_ref_plot_label = "Reference"
    elif problem == ne.InitialData.Sedov:
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
    elif problem == ne.InitialData.LaneEmden:
        x_ref = x
        star_mask = star_mask = np.abs(x) < np.pi - 1.0e-19

        mass_density_ref = np.full(len(x), 1.0e-20)
        mass_density_ref[star_mask] = np.abs(
            np.sin(x[star_mask]) / x[star_mask])

        # polytropic constant K = 1, and rho_c = 1
        pressure_ref = mass_density_ref**2
        velocity_ref = np.full(len(x), 0.0)
        exact_or_ref_plot_label = "Exact"
    else:
        import matplotlib.pyplot as plt
        plt.plot(x, mass_density, 'o')
        plt.title("t=%1.3e" % (time))
        plt.yscale('log')
        plt.show()
        return None

    plot.generate_plot_with_reference(
        x,
        x_ref,
        mass_density,
        mass_density_ref,
        "Density",
        str(problem).replace("InitialData.", '') + str(num_cells) +
        "Density.pdf",
        exact_or_ref_plot_label,
        every_n,
        set_log_y=(problem == ne.InitialData.LeBlanc
                   or problem == ne.InitialData.LaneEmden))
    plot.generate_plot_with_reference(
        x,
        x_ref,
        ne.compute_pressure(mass_density, momentum_density, energy_density),
        pressure_ref,
        "Pressure",
        str(problem).replace("InitialData.", '') + str(num_cells) +
        "Pressure.pdf",
        exact_or_ref_plot_label,
        every_n,
        set_log_y=(problem == ne.InitialData.LeBlanc
                   or problem == ne.InitialData.LaneEmden))
    plot.generate_plot_with_reference(
        x, x_ref, momentum_density / mass_density, velocity_ref, "Velocity",
        str(problem).replace("InitialData.", '') + str(num_cells) +
        "Velocity.pdf", exact_or_ref_plot_label, every_n)
    plot.generate_plot_with_reference(
        x, x_ref, order_used, None, "Local Order",
        str(problem).replace("InitialData.", '') + str(num_cells) +
        "Order.pdf", exact_or_ref_plot_label, every_n)

    if generate_spacetime_plots:
        spacetime_history[-1][spacetime_history[-1] > 10] = 9
        plot.generate_spacetime_plot(str(problem).replace("InitialData.", '') +
                                     str(num_cells) + "SpacetimeDensity.pdf",
                                     spacetime_history[0],
                                     r"$\rho$",
                                     x,
                                     times,
                                     smoothen=True,
                                     set_log_y=True)
        plot.generate_spacetime_plot(str(problem).replace("InitialData.", '') +
                                     str(num_cells) + "SpacetimeOrder.pdf",
                                     spacetime_history[-1],
                                     "Order",
                                     x,
                                     times,
                                     smoothen=False,
                                     set_log_y=False)

        if problem == ne.InitialData.LaneEmden:
            for i in range(spacetime_history[0].shape[0]):
                spacetime_history[0][i] = (
                    np.abs(spacetime_history[0][i] - mass_density_ref) /
                    mass_density_ref + 1.0e-20)

            plot.generate_spacetime_plot(
                str(problem).replace("InitialData.", '') + str(num_cells) +
                "SpacetimeDensityError.pdf",
                spacetime_history[0],
                r"$|\rho-\rho_{\mathrm{exact}}|$",
                x,
                times,
                vmin=-7,
                vmax=0,
                smoothen=True,
                set_log_y=True)


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
        "  Interacting blast waves: 400\n"
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
    parser.add_argument('--spacetime-plot',
                        dest='spacetime_plot',
                        action='store_true')
    parser.add_argument('--no-spacetime-plot',
                        dest='spacetime_plot',
                        action='store_false')
    parser.set_defaults(spacetime_plot=False)
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    main(args['problem'], args['number_of_cells'], args['numerical_flux'],
         args['cfl'], args['spacetime_plot'])
