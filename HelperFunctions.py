import numpy as np
from BoundaryClasses import BoundaryCondition, BoundaryParticleInteraction, BoundaryTypes


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
        values = dynamic_values_function(t=0)
    elif side == "RIGHT":
        xs = np.zeros(num_y_nodes) + num_x_nodes - 1
        ys = np.arange(num_y_nodes)
        values = dynamic_values_function(t=0)
    elif side == "LOWER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes)
        values = dynamic_values_function(t=0)
    elif side == "UPPER":
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes) + num_y_nodes - 1
        values = dynamic_values_function(t=0)

    edge_boundary = BoundaryCondition(bc_type, np.array([xs.astype(int), ys.astype(int)]), dynamic_values_function)

    return edge_boundary


# FUNCTIONS FOR TESTING DYNAMIC BOUNDARY CONDITIONS
def sinusoidal(amplitude, period, phase, t):
    return np.sin(2*np.pi*(t/period) + phase) * amplitude


# GENERATE FROM DISTRIBUTION
def uniform_side_flux(side, x_length, y_length, delta_x, delta_y):
    # TODO: Should this be initialised randomly within cell width, or within v*dt?
    x_length -= 1e-6
    y_length -= 1e-6
    rand = np.random.rand()
    if side == "LEFT":
        position = np.array([[rand*delta_x], [rand*y_length]])
    elif side == "RIGHT":
        position = np.array([[x_length - rand*delta_x], [rand*y_length]])
    elif side == "LOWER":
        position = np.array([[rand*x_length], [rand*delta_y]])
    elif side == "UPPER":
        position = np.array([[rand*x_length], [y_length - rand*delta_y]])
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
