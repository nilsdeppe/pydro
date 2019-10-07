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
problem = "strong"
iterations = 400
num_cells = 400
time_order = 2

ne.set_symmetry(ne.Symmetry.No)
ne.set_numerical_flux(ne.NumericalFlux.Hll)
reconstruct_prims = True

initial_position = 0.5

PR = 4000.0
xmin = 0.0
xmax = 1.0
tmax = 0.2
cfl = 0.05
# End configuration
################################################################

num_faces = num_cells + 1
num_interfaces = 2 * (num_faces)
plt.clf()
dx = (xmax - xmin) / num_cells
x = xmin + (np.arange(num_cells) + 0.5) * dx
x_face = xmin + (np.arange(num_cells + 1)) * dx
dt = cfl * dx

if problem == "sod":
    ne.set_gamma(1.4)
    left_mass = 1.0
    left_velocity = 0.0
    left_pressure = 1.0
    right_mass = 0.125
    right_velocity = 0.0
    right_pressure = 0.1
elif problem == "lax":
    ne.set_gamma(1.4)
    left_mass = 0.445
    left_velocity = 0.698
    left_pressure = 3.528
    right_mass = 0.5
    right_velocity = 0.0
    right_pressure = 0.571
elif problem == "strong":
    ne.set_gamma(1.4)
    # https://iopscience.iop.org/article/10.1086/317361
    left_mass = 10.0
    left_velocity = 0.0
    left_pressure = 100.0
    right_mass = 1.0
    right_velocity = 0.0
    right_pressure = 1.0
elif problem == "123":
    ne.set_gamma(1.4)
    left_mass = 1.0
    left_velocity = -2.0
    left_pressure = 0.4
    right_mass = 1.0
    right_velocity = 2.0
    right_pressure = 0.4
elif problem == "severe shock":
    ne.set_gamma(1.4)
    left_mass = 1.0
    left_velocity = 0.0
    left_pressure = 0.1 * PR
    right_mass = 1.0
    right_velocity = 0.0
    right_pressure = 0.1

left_momentum_density = left_mass * left_velocity
right_momentum_density = right_mass * right_velocity
left_energy_density = ne.compute_energy_density(left_mass,
                                                left_momentum_density,
                                                left_pressure)
right_energy_density = ne.compute_energy_density(right_mass,
                                                 right_momentum_density,
                                                 right_pressure)


def take_step(reconsstruction_scheme, deriv_scheme, history, mass_density,
              momentum_density, energy_density, dt):
    order_used = np.zeros(len(mass_density), dtype=int) + 100
    if reconstruct_prims:
        recons_mass, recons_velocity, recons_energy = recons.reconstruct([
            mass_density, momentum_density / mass_density,
            energy_density / mass_density
        ], reconsstruction_scheme, order_used)
        recons_momentum = recons_velocity * recons_mass
        recons_energy *= recons_mass
    else:
        recons_mass, recons_momentum, recons_energy = recons.reconstruct(
            [mass_density, momentum_density, energy_density],
            reconsstruction_scheme, order_used)

    bc_distance = 5
    recons_mass[0:bc_distance] = left_mass
    recons_mass[-bc_distance:] = right_mass
    recons_momentum[0:bc_distance] = left_momentum_density
    recons_momentum[-bc_distance:] = right_momentum_density
    recons_energy[0:bc_distance] = left_energy_density
    recons_energy[-bc_distance:] = right_energy_density

    recons_mass_flux, recons_momentum_flux, recons_energy_flux = ne.compute_flux(
        [recons_mass, recons_momentum, recons_energy])
    nf_mass, nf_momentum, nf_energy = ne.compute_numerical_flux(
        recons_mass_flux, recons_momentum_flux, recons_energy_flux,
        recons_mass, recons_momentum, recons_energy)

    dmass_dt, dmomentum_dt, denergy_dt = Derivative.differentiate_flux(
        deriv_scheme, dx, [nf_mass, nf_momentum, nf_energy],
        ne.compute_flux([mass_density, momentum_density, energy_density])
        if deriv_scheme == Derivative.Scheme.MND4
        or deriv_scheme == Derivative.Scheme.MND6
        or deriv_scheme == Derivative.Scheme.MNDV else None, order_used)
    dmass_dt *= -1
    dmomentum_dt *= -1
    denergy_dt *= -1

    if ne.get_symmetry() != 0:
        mass_source, momentum_source, energy_source = ne.compute_sources(
            x, [mass_density, momentum_density, energy_density])
        dmass_dt = dmass_dt + mass_source
        dmomentum_dt = dmomentum_dt + momentum_source
        denergy_dt = denergy_dt + energy_source

    return TimeStepper.adams_bashforth(
        time_order, history, [mass_density, momentum_density, energy_density],
        [dmass_dt, dmomentum_dt, denergy_dt], dt)


def set_initial_data():
    time = 0.0
    mass_density = np.full(len(x), left_mass)
    mass_density[x > initial_position] = right_mass

    momentum_density = np.full(len(x), left_momentum_density)
    momentum_density[x > initial_position] = right_momentum_density

    energy_density = np.full(len(x), left_energy_density)
    energy_density[x > initial_position] = right_energy_density
    return (time, mass_density, momentum_density, energy_density)


frequency = iterations / 10


def do_solve(reconsstruction_scheme, deriv_scheme):
    history = []
    time, mass_density, momentum_density, energy_density = set_initial_data()
    for i in range(iterations):
        time += dt
        mass_density, momentum_density, energy_density = take_step(
            reconsstruction_scheme, deriv_scheme, history, mass_density,
            momentum_density, energy_density, dt)
        # if i % frequency == 0:
        #     c = 1.0 - (0.1 + (i / frequency) * 0.1)
        #     plt.plot(x, mass_density, ls=":", color=str(c), zorder=-1)

    return (time, mass_density, momentum_density, energy_density)


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
