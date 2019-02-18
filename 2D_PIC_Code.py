import numpy as np
import matplotlib.colors as colours
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as time
from functools import partial

from ParticleManager import ParticleManager
from ParticleSource import ParticleSource
from Diagnostics import EnergyDiagnostic, ParticleSystemHistory
from HelperFunctions import*
from BoundaryClasses import*
from ParticleTypes import*

# THIS CODE IS 54% RILEY APPROVED


# TODO: Save animation to video
# ANIMATION
def animate_step(i):
    particle_system.update_step()

    # animate particles
    # positions.set_data(particle_system.particle_positions[0, :], particle_system.particle_positions[1, :])

    # animate charge density
    densities.set_data(particle_system.grid_charge_densities.T[::-1])

    # animate potentials
    potentials.set_array(particle_system.grid_potentials.T[::-1])

    return potentials, densities,


def animate_history(i):
    # animate charge density
    densities.set_data(particle_system_history.grid_charge_densities_history[i].T)

    # set time text
    timetext.set_text(f'{particle_system_history.simulation_time_history[i]*1000:<6.4f}ms')

    # animate potentials
    potentials.set_array(particle_system_history.grid_potentials_history[i].T)

    return potentials, densities, timetext


# SIMULATION SETTINGS
animate_live = False  # animate simulation as it is running (VERY SLOW)
animate_at_end = True  # animate simulation at end
printout_interval = 500  # timesteps between printouts
snapshot_interval = 2  # timesteps between snapshots
num_time_steps = 2000  # num of time steps to simulate for

# INITIALISE PARTICLES
def initialise_test():
    # PARAMETERS
    eps_0 = 8.854e-12  # permittivity of free space
    delta_t = 1e-6  # step size
    num_x_nodes = 150
    num_y_nodes = 100
    x_length = 0.015
    y_length = 0.01
    delta_x = x_length / (num_x_nodes - 1)  # grid resolution
    delta_y = y_length / (num_y_nodes - 1)  # grid resolution

    num_particles = 500  # keep this as an even number for now

    # particle_positions = np.array([[], []])
    # particle_velocities = np.array([[], []])
    # particle_charges = np.array([])
    # particle_masses = np.array([])

    particle_positions = np.random.rand(2, num_particles) * [[x_length], [y_length]]
    particle_velocities = (np.random.rand(2, num_particles) - 0.5)
    particle_charges = np.append(np.zeros(int(num_particles / 2)) + ArgonIon().charge,
                                 np.zeros(int(num_particles / 2)) + Electron().charge)
    particle_masses = np.append(np.zeros(int(num_particles / 2)) + ArgonIon().mass,
                                np.zeros(int(num_particles / 2)) + Electron().mass)
    particle_types = np.append([ArgonIon().type]*int(num_particles/2), [Electron().type]*int(num_particles/2))

    particle_sources = []

    left_bc = LeftBoundary(num_x_nodes, num_y_nodes, BoundaryTypes.FLOATING, 0,
                           BoundaryParticleInteraction.DESTROY, collect_charge=True)
    right_bc = RightBoundary(num_x_nodes, num_y_nodes, BoundaryTypes.FLOATING, 0,
                            BoundaryParticleInteraction.DESTROY, collect_charge=True)
    upper_bc = UpperBoundary(num_x_nodes, num_y_nodes, BoundaryTypes.NEUMANN, 0,
                            BoundaryParticleInteraction.REFLECT, collect_charge=False)
    lower_bc = LowerBoundary(num_x_nodes, num_y_nodes, BoundaryTypes.NEUMANN, 0,
                            BoundaryParticleInteraction.REFLECT, collect_charge=False)

    bcs = [left_bc, right_bc, upper_bc, lower_bc]

    return delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, particle_positions, particle_velocities, particle_charges, particle_masses, particle_types, bcs, particle_sources


def test_2():
    # Constants
    EPS_0 = 8.854e-12  # permittivity of free space, F/m
    QE = 1.602e-19  # elementary charge, C
    K = 1.381e-23  # boltzmann constant, m2kgs-2K-1
    MI = 32*1.661e-27  # ion mass, kg
    ME = 9.109e-31  # electron mass, kg

    # Sim settings
    n0 = 1e12  # neutral density, 1/m3
    phi0 = 0  # reference potential, V
    Te = 1  # electron temperature, eV
    Ti = 0.1  # ion velocity, eV
    v_drift = 7000  # ion injection velocity, m/s
    phi_p = -5  # plate potential, V

    # Calc plasma parameters
    lD = np.sqrt(EPS_0*Te / (n0*QE))  # Debye length, m
    vth = np.sqrt(2*QE*Ti/MI)  # Thermal velocity of ions, eV

    # Sim domain
    nx = 16  # num x nodes
    ny = 10  # num y nodes
    ts = 200  # num time steps
    dh = lD  # cell width / height
    np_insert = (ny-1)*15  # number of particles to insert per cell

    # Other values
    nn = nx*ny  # total number of nodes
    dt = 0.1*dh/v_drift  # time step, s, cross 0.1dh per timestep
    Lx = (nx-1)*dh  # x domain length
    Ly = (ny-1)*dh  # y domain length

    # Plate dimensions
    lower_left = [np.floor(nx/3).astype(int), 0]
    upper_right = [np.floor(nx/3).astype(int)+2, np.floor(ny/2).astype(int)]

    # Calc specific weight
    flux = n0*v_drift*Ly  # flux of entering particles
    npt = flux*dt  # number of real particles created per time step
    spwt = npt/np_insert  # ratio of real particles per macroparticle
    mp_q = 1  # macroparticle charge

    # Boundary conditions
    left_bc = create_uniform_edge_boundary(nx, ny, "LEFT", BoundaryTypes.NEUMANN, 0)
    right_bc = create_uniform_edge_boundary(nx, ny, "RIGHT", BoundaryTypes.NEUMANN, 0)
    upper_bc = create_uniform_edge_boundary(nx, ny, "UPPER", BoundaryTypes.NEUMANN, 0)
    lower_bc = create_uniform_edge_boundary(nx, ny, "LOWER", BoundaryTypes.DIRICHLET, 0)
    charged_plate = create_charged_plate(lower_left, upper_right, -7, BoundaryTypes.DIRICHLET)

    bcs = [left_bc, right_bc, upper_bc, lower_bc, charged_plate]

    left_boundary_interaction = BoundaryParticleInteraction.OPEN
    right_boundary_interaction = BoundaryParticleInteraction.OPEN
    upper_boundary_interaction = BoundaryParticleInteraction.OPEN
    lower_boundary_interaction = BoundaryParticleInteraction.REFLECTIVE

    particle_sources = []

    return (dt, EPS_0,
            nx, ny, Lx, Ly, bcs, particle_sources,
            left_boundary_interaction, right_boundary_interaction, upper_boundary_interaction, lower_boundary_interaction)


def test_3():
    eps_0 = 8.854e-12  # permittivity of free space, F/m
    e = 1.602e-19  # elementary charge, C
    k = 1.381e-23  # boltzmann constant, m2kgs-2K-1
    mi = 32 * 1.661e-27  # ion mass, kg
    me = 9.109e-31  # electron mass, kg

    dt = 1e-6

    x_length = 0.015
    y_length = 0.0001
    num_x_nodes = 150
    num_y_nodes = 1

    num_particles = 10

    particle_positions = np.random.rand(2, num_particles) * [[x_length], [y_length]]
    particle_velocities = (np.random.rand(2, num_particles) - 0.5)
    particle_charges = np.append(np.zeros(int(num_particles / 2)) + e,
                                 np.zeros(int(num_particles / 2)) - e)
    particle_masses = np.append(np.zeros(int(num_particles / 2)) + mi,
                                np.zeros(int(num_particles / 2)) + me)






    left_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", BoundaryTypes.DIRICHLET, 0,
                                           BoundaryParticleInteraction.REFLECT)
    right_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", BoundaryTypes.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)
    upper_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", BoundaryTypes.NEUMANN, 0,
                                            BoundaryParticleInteraction.REFLECT)
    lower_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", BoundaryTypes.NEUMANN, 0,
                                            BoundaryParticleInteraction.REFLECT)

    bcs = [left_bc, right_bc, upper_bc, lower_bc]


    return dt, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, \
           particle_positions, particle_velocities, particle_charges, particle_masses, bcs


def pretty_demo():
    # PARAMETERS
    eps_0 = 1  # permittivity of free space
    delta_t = 0.05  # step size
    num_x_nodes = 150
    num_y_nodes = 100
    x_length = 150
    y_length = 100
    delta_x = x_length / (num_x_nodes - 1)  # grid resolution
    delta_y = y_length / (num_y_nodes - 1)  # grid resolution

    num_particles = 10000  # keep this as an even number for now

    particle_positions = np.array([[], []])
    particle_velocities = np.array([[], []])
    particle_charges = np.array([])
    particle_masses = np.array([])

    # particle_positions = np.random.rand(2, num_particles) * [[x_length], [y_length]]
    # particle_velocities = (np.random.rand(2, num_particles) - 0.5) * 10
    # particle_charges = np.append(np.zeros(int(num_particles / 2)) + 10,
    #                              np.zeros(int(num_particles / 2)) - 10)
    # particle_masses = np.append(np.zeros(int(num_particles / 2)) + 1,
    #                             np.zeros(int(num_particles / 2)) + 1)

    ion_input = ParticleSource(
        partial(uniform_side_flux, side="LEFT", x_length=x_length, y_length=y_length, delta_x=delta_x, delta_y=delta_y, v_drift=10, delta_t=delta_t),
        [[10], [0]],
        [1],
        [10],
        100)

    electron_input = ParticleSource(
        partial(uniform_side_flux, side="LEFT", x_length=x_length, y_length=y_length, delta_x=delta_x, delta_y=delta_y, v_drift=10, delta_t=delta_t),
        [[10], [0]],
        [1],
        [-10],
        100)

    particle_sources = [ion_input, electron_input]

    left_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", BoundaryTypes.NEUMANN, 0,
                                           BoundaryParticleInteraction.REFLECT)
    right_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", BoundaryTypes.NEUMANN, 0,
                                            BoundaryParticleInteraction.DESTROY)
    upper_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", BoundaryTypes.DIRICHLET,0,
                                            BoundaryParticleInteraction.REFLECT)
    lower_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", BoundaryTypes.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)

    charged_plate = create_charged_plate([int(num_x_nodes / 3), 0], [int(num_x_nodes * 2 / 3), int(num_y_nodes / 4)], partial(sinusoidal, amplitude=20, period=20, phase=0),
                                         BoundaryTypes.DIRICHLET, BoundaryParticleInteraction.DESTROY)

    charged_plate2 = create_charged_plate([int(num_x_nodes / 3), int(num_y_nodes * 3 / 4)], [int(num_x_nodes * 2 / 3), int(num_y_nodes-1)], partial(sinusoidal, amplitude=20, period=20, phase=np.pi),
                                         BoundaryTypes.DIRICHLET, BoundaryParticleInteraction.DESTROY)

    bcs = [left_bc, right_bc, upper_bc, lower_bc, charged_plate, charged_plate2]

    return delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, particle_positions, particle_velocities, particle_charges, particle_masses, bcs, particle_sources

def pretty_demo2():
    # PARAMETERS
    eps_0 = 1  # permittivity of free space
    delta_t = 0.05  # step size
    num_x_nodes = 150
    num_y_nodes = 100
    x_length = 150
    y_length = 100
    delta_x = x_length / (num_x_nodes - 1)  # grid resolution
    delta_y = y_length / (num_y_nodes - 1)  # grid resolution

    num_particles = 10000  # keep this as an even number for now

    particle_positions = np.array([[], []])
    particle_velocities = np.array([[], []])
    particle_charges = np.array([])
    particle_masses = np.array([])

    # particle_positions = np.random.rand(2, num_particles) * [[x_length], [y_length]]
    # particle_velocities = (np.random.rand(2, num_particles) - 0.5) * 10
    # particle_charges = np.append(np.zeros(int(num_particles / 2)) + 10,
    #                              np.zeros(int(num_particles / 2)) - 10)
    # particle_masses = np.append(np.zeros(int(num_particles / 2)) + 1,
    #                             np.zeros(int(num_particles / 2)) + 1)

    ion_input = ParticleSource(
        partial(uniform_side_flux, side="LEFT", x_length=x_length, y_length=y_length, delta_x=delta_x, delta_y=delta_y, v_drift=10, delta_t=delta_t),
        [[10], [0]],
        [1],
        [10],
        100)

    electron_input = ParticleSource(
        partial(uniform_side_flux, side="LEFT", x_length=x_length, y_length=y_length, delta_x=delta_x, delta_y=delta_y, v_drift=10, delta_t=delta_t),
        [[10], [0]],
        [0.1],
        [-10],
        1000)

    particle_sources = [ion_input, electron_input]

    left_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", BoundaryTypes.DIRICHLET, 0,
                                           BoundaryParticleInteraction.REFLECT)
    right_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", BoundaryTypes.DIRICHLET, 0,
                                            BoundaryParticleInteraction.DESTROY)
    upper_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", BoundaryTypes.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)
    lower_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", BoundaryTypes.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)

    bcs = [left_bc, right_bc, upper_bc, lower_bc]

    return delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, particle_positions, particle_velocities, particle_charges, particle_masses, bcs, particle_sources

# CREATE PARTICLE SYSTEM

# delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, \
# particle_positions, particle_velocities, particle_charges, particle_masses, \
# bcs, particle_sources = initialise_test()

delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, particle_positions, particle_velocities, particle_charges, particle_masses, particle_types, bcs, particle_sources = initialise_test()

particle_system = ParticleManager(delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length,
                                  particle_positions, particle_velocities, particle_charges, particle_masses, particle_types,
                                  boundary_conditions=bcs, particle_sources=particle_sources, replace_lost_particles=True)

particle_system_history = ParticleSystemHistory(particle_system)
energy_diagnostic = EnergyDiagnostic()


if animate_live:
    animation_object = animation.FuncAnimation(fig, animate_step, interval=1, blit=True)
    plt.show()
else:
    for i in range(num_time_steps):
        particle_system.update_step()

        # printout system state
        if particle_system.current_t_step % printout_interval == 0:
            print(f'{particle_system.current_t_step/num_time_steps*100:<6.2f}% completion')
            print("Computational Time Elapsed =", time.time() - particle_system.start_time, "s")
            average_time_per_step = (time.time() - particle_system.start_time) / particle_system.current_t_step
            steps_to_go = num_time_steps - particle_system.current_t_step
            approx_time_to_go = steps_to_go*average_time_per_step
            print("Approx Time To Finish =", round(approx_time_to_go), "s")
            print("T Step =", particle_system.current_t_step, "/", num_time_steps)
            print("Simulation Time =", particle_system.current_t_step*particle_system.delta_t, "s")
            print("Total Energy =", energy_diagnostic.energy_total_history[-1])
            print("Num Particles =", particle_system.num_particles)
            print(" ")

        # take a system snapshot for animation
        if particle_system.current_t_step % snapshot_interval == 0:
            particle_system_history.take_snapshot()
            energy_diagnostic.calc_system_energy(particle_system)

print("Simulation finished!")
#input("Press ENTER to play animation")

if animate_at_end:  # run animation
    # SET UP FIGURES
    fig = plt.figure(figsize=(6, 8))

    ax1 = fig.add_subplot(211)
    ax1.set_title("Densities")
    densities = ax1.imshow(np.zeros((particle_system.num_x_nodes, particle_system.num_y_nodes)),
                           vmin=np.min([np.min(i) for i in particle_system_history.grid_charge_densities_history]),
                           vmax=np.max([np.max(i) for i in particle_system_history.grid_charge_densities_history]),
                           interpolation="bicubic", origin="lower",
                           extent=[0, particle_system.x_length, 0, particle_system.y_length])

    ax2 = fig.add_subplot(212)
    ax2.set_title("Potential")
    potentials = ax2.imshow(np.zeros((particle_system.num_x_nodes, particle_system.num_y_nodes)),
                            vmin=np.min([np.min(i) for i in particle_system_history.grid_potentials_history]),
                            vmax=np.max([np.max(i) for i in particle_system_history.grid_potentials_history]),
                            interpolation="bicubic", origin="lower",
                            extent=[0, particle_system.x_length, 0, particle_system.y_length])

    ax3 = fig.add_subplot(211)
    timetext = ax3.text(particle_system.x_length / 2, particle_system.y_length * 0.9, "0")

    animation_object = animation.FuncAnimation(fig, animate_history, frames=len(particle_system_history.grid_potentials_history), save_count=len(particle_system_history.grid_potentials_history), interval=1, blit=True)
    #animation_object.save('video.mp4', writer="ffmpeg")
    plt.show()

# plot
energy_diagnostic.plot_energy_history()
particle_system_history.plot_average_potential()
particle_system_history.plot_particle_number()

