import numpy as np
from BoundaryClasses import BoundaryCondition, BoundaryParticleInteraction, BoundaryTypes, BoundaryLocations


# FUNCTIONS ALLOW FOR EASY BOUNDARY CONDITION CREATION
def create_charged_plate(lower_left_corner, upper_right_corner, value, bc_type, interaction, neumann_direction=None):
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

    charged_plate = BoundaryCondition(bc_type, np.array([xs.astype(int), ys.astype(int)]), values, BoundaryLocations.INTERIOR, neumann_direction=neumann_direction, particle_interaction=interaction)

    return charged_plate


def create_uniform_edge_boundary(num_x_nodes, num_y_nodes, side, bc_type, value, interaction):
    neumann_direction = None
    if side == "LEFT":
        xs = np.zeros(num_y_nodes)
        ys = np.arange(num_y_nodes)
        location = BoundaryLocations.LEFT
        if bc_type == BoundaryTypes.NEUMANN:
            neumann_direction = 0
    elif side == "RIGHT":
        xs = np.zeros(num_y_nodes) + num_x_nodes - 1
        ys = np.arange(num_y_nodes)
        location = BoundaryLocations.RIGHT
        if bc_type == BoundaryTypes.NEUMANN:
            neumann_direction = 0
    elif side == "LOWER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes)
        location = BoundaryLocations.LOWER
        if bc_type == BoundaryTypes.NEUMANN:
            neumann_direction = 1
    elif side == "UPPER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes) + num_y_nodes - 1
        location = BoundaryLocations.UPPER
        if bc_type == BoundaryTypes.NEUMANN:
            neumann_direction = 1

    edge_boundary = BoundaryCondition(bc_type, np.array([xs.astype(int), ys.astype(int)]), value, location, neumann_direction=neumann_direction, particle_interaction=interaction)

    return edge_boundary

# FUNCTIONS FOR TESTING DYNAMIC BOUNDARY CONDITIONS
def sinusoidal(amplitude, period, phase, t):
    return np.sin(2*np.pi*(t/period) + phase) * amplitude


# GENERATE FROM DISTRIBUTION
def uniform_side_flux(side, x_length, y_length, delta_x, delta_y, v_drift, delta_t):
    x_length -= 1e-6
    y_length -= 1e-6
    rand = np.random.rand()
    if side == "LEFT":
        position = np.array([[rand*v_drift*delta_t], [rand*y_length]])
    elif side == "RIGHT":
        position = np.array([[x_length - rand*v_drift*delta_t], [rand*y_length]])
    elif side == "LOWER":
        position = np.array([[rand*x_length], [rand*v_drift*delta_t]])
    elif side == "UPPER":
        position = np.array([[rand*x_length], [y_length - rand*v_drift*delta_t]])
    else:
        raise ValueError("Please specify side")
    return position


def sample_maxwell_boltzmann_velocity_distribution(v_thermal, M):
    # TODO: Is this right?
    a = 0
    for i in range(M):
        a += np.random.rand()
    velocity = np.sqrt(M/12) * (a - M/2) * v_thermal

    return velocity


def new_velocity(v_drift, v_thermal, M):
    x_vel = sample_maxwell_boltzmann_velocity_distribution(v_thermal, M)
    y_vel = sample_maxwell_boltzmann_velocity_distribution(v_thermal, M)

    velocity = np.array([[x_vel], [y_vel]]) + v_drift

    return velocity
