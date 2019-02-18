import numpy as np
from BoundaryClasses import BoundaryCondition, BoundaryParticleInteraction, BoundaryTypes, BoundaryLocations

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
