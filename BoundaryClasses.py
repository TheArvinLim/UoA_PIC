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

    """Object that stores the type and value of a given boundary condition"""
    def __init__(self, type, positions, static_values=None, dynamic_values_function=None, neumann_direction=None):
        """
        :param type: The boundary condition type as a BoundaryTypes object
        :param value: The value(s) of this boundary condition as a 1D array
        :param static_values: node coordinates of this boundary
        :param dynamic_values_function: node coordinates of this boundary
        :param Neumann_direction: if type is Neumann, specify whether in x (0) or y (1) direction
        """
        self.positions = positions
        self.type = type
        self.dynamic_values_function = dynamic_values_function

        if self.type == BoundaryTypes.NEUMANN:
            if neumann_direction is None:
                raise ValueError('Please specify direction of Neumann boundary condition')
            elif not (neumann_direction==0 or neumann_direction==1):
                raise ValueError('Please specify Neumann boundary condition as either in x direction (0) or y direction (1)')
            else:
                self.neumann_direction = neumann_direction

        if dynamic_values_function is not None:
            self.values = dynamic_values_function(0)

        elif static_values is not None:
            self.values = static_values

        else:
            raise ValueError('Value was not set for this boundary condition')


    def update(self, t):
        self.values = self.dynamic_values_function(t)