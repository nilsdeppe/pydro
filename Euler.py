# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import time as time_mod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick

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

# Use standard LaTeX font on plots
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

################################################################
# Configuration:
reconstruct_prims = True
# End configuration
################################################################


def time_deriv(stepper, evolved_vars, time):
    if reconstruct_prims:
        primitive_vars = None
        primitive_vars = ne.compute_primitives(primitive_vars, evolved_vars)
        recons_prim = stepper.reconstruct_variables(primitive_vars)
        recons_conserved = ne.compute_conserved(recons_prim)
    else:
        recons_conserved = stepper.reconstruct_variables(evolved_vars)

    numerical_fluxes_at_faces = ne.compute_numerical_flux(recons_conserved)

    dt_evolved_vars = -1.0 * stepper.flux_deriv(numerical_fluxes_at_faces,
                                                ne.compute_flux(evolved_vars))

    if ne.get_symmetry() != 0:
        dt_evolved_vars += ne.compute_sources(stepper.get_x(), evolved_vars)

    bc_distance = 7
    # zero time derivs at boundary
    for i in range(len(dt_evolved_vars)):
        dt_evolved_vars[i][0:bc_distance] = 0.0
        dt_evolved_vars[i][-bc_distance:] = 0.0
    return dt_evolved_vars


def do_solve(num_cells, problem, cfl, reconstructor, reconstruction_scheme,
             deriv_scheme):
    time, final_time, x, mass_density, momentum_density, energy_density = ne.set_initial_data(
        num_cells, problem)

    stepper = TimeStepper.Rk3Ssp(
        time_deriv, x, reconstructor, reconstruction_scheme, deriv_scheme,
        np.asarray([mass_density, momentum_density, energy_density]), time)

    while stepper.get_time() <= final_time:
        mass_density, momentum_density, energy_density = stepper.get_evolved_vars(
        )
        sound_speed = ne.compute_sound_speed(
            mass_density,
            ne.compute_pressure(mass_density, momentum_density,
                                energy_density))
        speed = np.amax(np.abs(momentum_density / mass_density) + sound_speed)
        dt = cfl * stepper.get_dx() / speed
        if stepper.get_time() + dt > final_time:
            dt = final_time - stepper.get_time()
            stepper.take_step(dt)
            break
        stepper.take_step(dt)

    mass_density, momentum_density, energy_density = stepper.get_evolved_vars()
    return (stepper.get_time(), x, stepper.get_order_used(), mass_density,
            momentum_density, energy_density)


def generate_plot_with_reference(x,
                                 x_ref,
                                 func,
                                 reference,
                                 quantity_name,
                                 file_name,
                                 ref_label,
                                 every_n=0,
                                 set_log_y=False):
    # We force the formatting so that all rendered images are the same size.
    # pyplot will change the size of the plot depending on how many significant
    # digits are shown...
    class ScalarFormatterForceFormat(mtick.ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

    if every_n == 0:
        every_n = len(x) // 50
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.set_size_inches(6, 6, forward=True)
    markersize = 5
    fontsize = 20
    linewidth = 1.8

    # Use a colorblind and grey scale friendly color palette.
    ax.set_prop_cycle(
        plt.cycler(color=['#0F2080', '#F5793A', '#A95AA1', '#85C0F9']))

    ax.plot(x[0::every_n],
            func[0::every_n],
            'o',
            markersize=markersize,
            label="Numerical")
    if quantity_name != "Local Order":
        ax.plot(x_ref, reference, '-', linewidth=linewidth, label=ref_label)

    ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel(r"$x$", fontsize=fontsize)
    ax.set_ylabel(quantity_name, fontsize=fontsize)

    plt.grid(b=True, which='major', linestyle='--')

    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0, 0))

    ax.yaxis.set_major_formatter(yfmt)
    ax.yaxis.offsetText.set_fontsize(fontsize - 4)

    ax.xaxis.set_major_formatter(ScalarFormatterForceFormat())
    ax.xaxis.offsetText.set_fontsize(fontsize - 4)

    if quantity_name != "Local Order":
        ax.legend(loc='best', fontsize=fontsize - 1)

    if set_log_y:
        plt.yscale('log')
    plt.savefig(file_name, transparent=True, format='pdf', bbox_inches='tight')


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
    if problem != ne.InitialData.ShuOsher and problem != ne.InitialData.Sedov:
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

    generate_plot_with_reference(x,
                                 x_ref,
                                 mass_density,
                                 mass_density_ref,
                                 "Density",
                                 str(problem).replace("InitialData.", '') +
                                 "Density" + str(num_cells) + ".pdf",
                                 exact_or_ref_plot_label,
                                 every_n,
                                 set_log_y=(problem == ne.InitialData.LeBlanc))
    generate_plot_with_reference(x,
                                 x_ref,
                                 ne.compute_pressure(mass_density,
                                                     momentum_density,
                                                     energy_density),
                                 pressure_ref,
                                 "Pressure",
                                 str(problem).replace("InitialData.", '') +
                                 "Pressure" + str(num_cells) + ".pdf",
                                 exact_or_ref_plot_label,
                                 every_n,
                                 set_log_y=(problem == ne.InitialData.LeBlanc))
    generate_plot_with_reference(
        x, x_ref, momentum_density / mass_density, velocity_ref, "Velocity",
        str(problem).replace("InitialData.", '') + "Velocity" +
        str(num_cells) + ".pdf", exact_or_ref_plot_label, every_n)
    generate_plot_with_reference(
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
                        help='The number of grid points/cells to use on the '
                        'finite-difference grid.')
    parser.add_argument('--cfl', type=float, help='The CFL factor to use.')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    main(args['problem'], args['number_of_cells'], args['numerical_flux'],
         args['cfl'])
