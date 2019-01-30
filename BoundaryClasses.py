from enum import Enum


class BoundaryTypes(Enum):
    """Stores all possible boundary condition types"""
    NEUMANN = 0  # sets value at boundary to constant
    DIRICHLET = 1  # sets derivative at boundary to constant


class BoundaryParticleInteraction(Enum):
    """Stores all possible actions to perform on a particle that collides with a boundary"""
    OPEN = 0  # destroys the particle
    REFLECTIVE = 1  # reflects the particle


class BoundaryCondition:
    # TODO: Allow for periodic boundary conditions
    """Object that stores the type and value of a given boundary condition"""
    def __init__(self, type, positions, magnitude_function, neumann_direction=None):
        """
        :param type: The boundary condition type as a BoundaryTypes object
        :param positions: The position(s) of this boundary condition as a 1D array
        :param magnitude_function: Magnitude(s) of the bc. Can be either float or function of time f(t)
        :param neumann_direction: if type is Neumann, specify whether in x (0) or y (1) direction
        """
        self.positions = positions
        self.type = type

        # check for direction of the neumann boundary
        if self.type == BoundaryTypes.NEUMANN:
            if neumann_direction is None:
                raise ValueError('Please specify direction of Neumann boundary condition')
            elif not (neumann_direction == 0 or neumann_direction == 1):
                raise ValueError('Please specify Neumann boundary condition as either in '
                                 'x direction (0) or y direction (1)')
            else:
                self.neumann_direction = neumann_direction

        # check whether a function of time, or just static
        if callable(magnitude_function):
            self.magnitude_function = magnitude_function
            self.values = magnitude_function(t=0)
            self.dynamic = True
        else:
            self.values = magnitude_function
            self.dynamic = False

    def update(self, t):
        """This function updates the magnitude of this boundary condition."""
        self.values = self.magnitude_function(t=t)