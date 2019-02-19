import numpy as np
import matplotlib.colors as colours
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as time
from functools import partial

from ParticleManager import ParticleManager
from ParticleSource import ParticleSource
from Diagnostics import ParticleSystemHistory
from HelperFunctions import*
from BoundaryClasses import*
from ParticleTypes import*

# THIS CODE IS 54% RILEY APPROVED


# TODO: Save animation to video
# INITIALISE PARTICLES
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

    left_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", FieldBoundaryCondition.NEUMANN, 0,
                                           BoundaryParticleInteraction.REFLECT)
    right_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", FieldBoundaryCondition.NEUMANN, 0,
                                            BoundaryParticleInteraction.DESTROY)
    upper_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", FieldBoundaryCondition.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)
    lower_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", FieldBoundaryCondition.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)

    charged_plate = create_charged_plate([int(num_x_nodes / 3), 0], [int(num_x_nodes * 2 / 3), int(num_y_nodes / 4)], partial(sinusoidal, amplitude=20, period=20, phase=0),
                                         FieldBoundaryCondition.DIRICHLET, BoundaryParticleInteraction.DESTROY)

    charged_plate2 = create_charged_plate([int(num_x_nodes / 3), int(num_y_nodes * 3 / 4)], [int(num_x_nodes * 2 / 3), int(num_y_nodes-1)], partial(sinusoidal, amplitude=20, period=20, phase=np.pi),
                                          FieldBoundaryCondition.DIRICHLET, BoundaryParticleInteraction.DESTROY)

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

    left_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LEFT", FieldBoundaryCondition.DIRICHLET, 0,
                                           BoundaryParticleInteraction.REFLECT)
    right_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "RIGHT", FieldBoundaryCondition.DIRICHLET, 0,
                                            BoundaryParticleInteraction.DESTROY)
    upper_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "UPPER", FieldBoundaryCondition.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)
    lower_bc = create_uniform_edge_boundary(num_x_nodes, num_y_nodes, "LOWER", FieldBoundaryCondition.DIRICHLET, 0,
                                            BoundaryParticleInteraction.REFLECT)

    bcs = [left_bc, right_bc, upper_bc, lower_bc]

    return delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, particle_positions, particle_velocities, particle_charges, particle_masses, bcs, particle_sources

# CREATE PARTICLE SYSTEM

# delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length, \
# particle_positions, particle_velocities, particle_charges, particle_masses, \
# bcs, particle_sources = initialise_test()
