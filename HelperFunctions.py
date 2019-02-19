import numpy as np
from BoundaryClasses import BoundaryCondition, BoundaryParticleInteraction, FieldBoundaryCondition, BoundaryLocations

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



