# copyright Nils Deppe 2019
# (See accompanying file LICENSE.md or copy at http://boost.org/LICENSE_1_0.txt)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
use_numba = True
try:
    import numba as nb
except:
    use_numba = False

import Reconstruction as recons
import TimeStepper
import Derivative
import NewtonianEuler as ne

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

################################################################
# Configuration:
problem = ne.InitialData.Severe1
final_time = 0.1
num_cells = 400

ne.set_symmetry(ne.Symmetry.No)
ne.set_numerical_flux(ne.NumericalFlux.Hll)
reconstruct_prims = True

initial_position = 0.5

xmin = 0.0
xmax = 1.0
tmax = 0.2
cfl = 0.2
# End configuration
################################################################


def init_grid():
    global num_faces
    global num_interfaces
    global dx
    global x
    global x_face
    global dt
    num_faces = num_cells + 1
    num_interfaces = 2 * (num_faces)
    dx = (xmax - xmin) / num_cells
    print("Global dx: ", dx)
    x = xmin + (np.arange(num_cells) + 0.5) * dx
    x_face = xmin + (np.arange(num_cells + 1)) * dx


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

    bc_distance = 7
    # zero time derivs at boundary
    for i in range(len(dt_evolved_vars)):
        dt_evolved_vars[i][0:bc_distance] = 0.0
        dt_evolved_vars[i][-bc_distance:] = 0.0
    return dt_evolved_vars


def do_solve(reconstruction_scheme, deriv_scheme):
    init_grid()

    time, mass_density, momentum_density, energy_density = ne.set_initial_data(
        x, problem, initial_position)

    stepper = TimeStepper.Rk3Ssp(
        time_deriv, x, recons.reconstruct, reconstruction_scheme, deriv_scheme,
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
    global global_order_used
    global_order_used = stepper.get_order_used()
    return (stepper.get_time(), mass_density, momentum_density, energy_density)


print("Starting solves...")
reconstruct_prims = True
time, mass_density, momentum_density, energy_density = do_solve(
    recons.Scheme.Minmod, Derivative.Scheme.MD)
plt.plot(x, mass_density, 'o', label="Minmod")

# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns5z, Derivative.Scheme.MD4)
# plt.plot(x, mass_density, 'o', label="WCNS5z-Prim")
# reconstruct_prims = False
# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns5z, Derivative.Scheme.MD4)
# plt.plot(x, mass_density, 'v', label="WCNS5z-Cons")

reconstruct_prims = True
# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns3, Derivative.Scheme.MD)
# plt.plot(x, mass_density, 's', label="WENO3-Prim-MD")

# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns3, Derivative.Scheme.MD4)
# plt.plot(x, mass_density, 'v', label="WENO3-Prim-MD4")

time, mass_density, momentum_density, energy_density = do_solve(
    recons.Scheme.Wcns3, Derivative.Scheme.MND4)
plt.plot(x, mass_density, '^', label="WENO3-Prim-MND4")

# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns5z, Derivative.Scheme.MD)
# plt.plot(x, mass_density, '<', label="WCNS5z-Prim-MD")

# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns5z, Derivative.Scheme.MND4)
# plt.plot(x, mass_density, '>', label="WCNS5z-Prim-MND4")

# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns5z, Derivative.Scheme.MND6)
# plt.plot(x, mass_density, 'v', label="WCNS5z-Prim-MND6")

# time, mass_density, momentum_density, energy_density = do_solve(
#     recons.Scheme.Wcns5, Derivative.Scheme.MND4)
# plt.plot(x, mass_density, '>', label="WCNS5-Prim-MND4")

plt.title("t=%1.3e" % time)
plt.legend()
plt.show()
