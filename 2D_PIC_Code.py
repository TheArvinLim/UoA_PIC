import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as time

from ParticleManager import ParticleManager
from ParticleSource import ParticleSource
from BoundaryClasses import BoundaryCondition, BoundaryParticleInteraction, BoundaryTypes
from Diagnostics import EnergyDiagnostic, ParticleSystemHistory

# THIS CODE IS 54% RILEY APPROVED


# FUNCTIONS ALLOW FOR EASY BOUNDARY CONDITION CREATION
def create_charged_plate(lower_left_corner, upper_right_corner, value, bc_type, neumann_direction=None):
    xsize = upper_right_corner[0] - lower_left_corner[0] + 1
    ysize = upper_right_corner[1] - lower_left_corner[1] + 1

    xs = np.zeros((xsize * ysize))
    ys = np.zeros((xsize * ysize))
    values = np.zeros((xsize * ysize))

    for i in range(xsize):
        for j in range(ysize):
            xs[j + i * ysize] = i + lower_left_corner[0]
            ys[j + i * ysize] = j + lower_left_corner[1]
            values[j + i * ysize] = value

    charged_plate = BoundaryCondition(bc_type, np.array([xs.astype(int), ys.astype(int)]), values, neumann_direction=neumann_direction)

    return charged_plate


def create_uniform_edge_boundary(num_x_nodes, num_y_nodes, side, bc_type, value, neumann_direction=None):
    if side == "LEFT":
        xs = np.zeros(num_y_nodes)
        ys = np.arange(num_y_nodes)
        values = np.zeros(num_y_nodes) + value
    elif side == "RIGHT":
        xs = np.zeros(num_y_nodes) + num_x_nodes - 1
        ys = np.arange(num_y_nodes)
        values = np.zeros(num_y_nodes) + value
    elif side == "LOWER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes)
        values = np.zeros(num_x_nodes) + value
    elif side == "UPPER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes) + num_y_nodes - 1
        values = np.zeros(num_x_nodes) + value

    edge_boundary = BoundaryCondition(bc_type, np.array([xs.astype(int), ys.astype(int)]), values, neumann_direction=neumann_direction)

    return edge_boundary


def create_dynamic_edge_boundary(num_x_nodes, num_y_nodes, side, bc_type, dynamic_values_function):
    if side == "LEFT":
        xs = np.zeros(num_y_nodes)
        ys = np.arange(num_y_nodes)
        values = dynamic_values_function(0)
    elif side == "RIGHT":
        xs = np.zeros(num_y_nodes) + num_x_nodes - 1
        ys = np.arange(num_y_nodes)
        values = dynamic_values_function(0)
    elif side == "LOWER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes)
        values = dynamic_values_function(0)
    elif side == "UPPER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes) + num_y_nodes - 1
        values = dynamic_values_function(0)

    edge_boundary = BoundaryCondition(bc_type, np.array([xs.astype(int), ys.astype(int)]), dynamic_values_function)

    return edge_boundary


# FUNCTIONS FOR TESTING DYNAMIC BOUNDARY CONDITIONS
def cosinusoidal(t):
    return np.zeros(100)+np.sin(t + np.pi)*2


def sinusoidal(t):
    return np.zeros(100)+np.sin(t)*2


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
    # animate particles
    # positions.set_data(particle_system.particle_positions[0, :], particle_system.particle_positions[1, :])

    # animate charge density
    densities.set_data(particle_system_history.grid_charge_densities_history[i].T)

    # animate potentials
    potentials.set_array(particle_system_history.grid_potentials_history[i].T)

    return potentials, densities,


# SIMULATION SETTINGS
num_time_steps = 5000  # num of time steps to simulate for
animate_live = False  # animate simulation as it is running (VERY SLOW)
animate_at_end = True  # animate simulation at end
printout_interval = 500  # timesteps between printouts
snapshot_interval = 10  # timesteps between snapshots
eps_0 = 1  # permittivity of free space
delta_t = 0.05  # step size


# INITIALISE PARTICLES
def initialise_test():
    # PARAMETERS
    num_x_nodes = 10
    num_y_nodes = 10
    delta_x = 1  # grid resolution
    delta_y = 1  # grid resolution
    x_length = (num_x_nodes-1)*delta_x
    y_length = (num_y_nodes-1)*delta_y
    num_particles = 20

    # particle_positions = np.random.rand(2, num_particles)*[[x_length], [y_length]]
    # particle_velocities = np.random.rand(2, num_particles)*delta_x*10
    # particle_charges = np.append(np.zeros(int(num_particles/2))-10, np.zeros(int(num_particles/2))+10)
    # particle_masses = np.append(np.zeros(int(num_particles/2))+1, np.zeros(int(num_particles/2))+1)

    particle_positions = np.array([[], []])
    particle_velocities = np.array([[], []])
    particle_charges = np.array([])
    particle_masses = np.array([])

    test_vel = 1
    four_point_cross = ParticleSource(
        [[delta_x, x_length - delta_x, x_length / 3, x_length / 3 * 2],  # x pos
         [y_length / 3, y_length * 2 / 3, delta_y, y_length - delta_y]],  # y pos
        [[test_vel, -test_vel, 0, -0],  # x vel
         [0, 0, test_vel, -test_vel]],  # y vel
        [1, 1, 1, 1],  # mass
        [1, -1, -1, 1],  # charge
        1)  # freq

    positron_source = ParticleSource([[x_length/2], [y_length/3]], [[0], [0]], 1, 1, 0.5)
    electron_source = ParticleSource([[x_length/2], [y_length/3*2]], [[0], [0]], 1, -1, 0.5)

    particle_sources = [four_point_cross]

    left_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", BoundaryTypes.DIRICHLET, 0.5)
    right_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", BoundaryTypes.DIRICHLET, -0.5)
    upper_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", BoundaryTypes.DIRICHLET, 0.5)
    lower_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", BoundaryTypes.DIRICHLET, -0.5)

    # left_bc = create_dynamic_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", BoundaryTypes.DIRICHLET, sinusoidal)
    # right_bc = create_dynamic_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", BoundaryTypes.DIRICHLET, cosinusoidal)
    #upper_bc = create_dynamic_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", BoundaryTypes.DIRICHLET, cosinusoidal)
    #lower_bc = create_dynamic_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", BoundaryTypes.DIRICHLET, cosinusoidal)

    charged_plate = create_charged_plate([50, 50], [55, 55], 1, BoundaryTypes.DIRICHLET)

    bcs = [left_bc, right_bc, upper_bc, lower_bc]

    left_boundary_interaction = BoundaryParticleInteraction.OPEN
    right_boundary_interaction = BoundaryParticleInteraction.OPEN
    upper_boundary_interaction = BoundaryParticleInteraction.OPEN
    lower_boundary_interaction = BoundaryParticleInteraction.OPEN

    return (particle_positions, particle_velocities, particle_charges, particle_masses, delta_t, eps_0,
            num_x_nodes, num_y_nodes, delta_x, delta_y, bcs, particle_sources,
            left_boundary_interaction, right_boundary_interaction, upper_boundary_interaction, lower_boundary_interaction)


# CREATE PARTICLE SYSTEM
particle_system = ParticleManager(*initialise_test())
particle_system_history = ParticleSystemHistory()
energy_diagnostic = EnergyDiagnostic()

# SET UP FIGURES
fig = plt.figure(figsize=(6, 8))

ax1 = fig.add_subplot(211)
ax1.set_title("Densities")
densities = ax1.imshow(np.zeros((particle_system.num_x_nodes, particle_system.num_y_nodes)), vmin=-1, vmax=1,
                       interpolation="bicubic", origin="lower",
                       extent=[0, particle_system.x_length, 0, particle_system.y_length])

ax2 = fig.add_subplot(212)
ax2.set_title("Potential")
potentials = ax2.imshow(np.zeros((particle_system.num_x_nodes, particle_system.num_y_nodes)), vmin=-1, vmax=1,
                        interpolation="bicubic", origin="lower",
                        extent=[0, particle_system.x_length, 0, particle_system.y_length])

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
            print("Simulation Time =", particle_system.current_t_step*particle_system.dt, "s")
            print("Total Energy =", energy_diagnostic.energy_total_history[-1])
            print("Num Particles =", particle_system.num_particles)
            print(" ")

        # take a system snapshot for animation
        if particle_system.current_t_step % snapshot_interval == 0:
            particle_system_history.take_snapshot(particle_system)
            energy_diagnostic.calc_system_energy(particle_system)

print("Simulation finished!")
input("Press ENTER to play animation")

if animate_at_end:  # run animation
    animation_object = animation.FuncAnimation(fig, animate_history, frames=len(particle_system_history.grid_potentials_history), save_count=len(particle_system_history.grid_potentials_history), interval=1, blit=True)
    plt.show()

# plot energy
energy_diagnostic.plot_energy_history()
