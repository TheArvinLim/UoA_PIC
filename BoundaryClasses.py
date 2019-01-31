import numpy as np
from enum import Enum


class BoundaryTypes(Enum):
    """Stores all possible boundary condition types"""
    NEUMANN = 0  # sets value at boundary to constant
    DIRICHLET = 1  # sets derivative at boundary to constant


class BoundaryParticleInteraction(Enum):
    """Stores all possible actions to perform on a particle that collides with a boundary"""
    DESTROY = 0  # destroys the particle
    REFLECT = 1  # reflects the particle


class BoundaryLocations(Enum):
    """Stores all possible boundary locations"""
    LEFT = 0
    RIGHT = 1
    UPPER = 2
    LOWER = 3
    INTERIOR = 4


class BoundaryCondition:
    # TODO: Allow for periodic boundary conditions
    """Object that stores the type and value of a given boundary condition"""
    def __init__(self, type, positions, magnitude_function, location, neumann_direction=None, particle_interaction=BoundaryParticleInteraction.DESTROY):
        """
        :param type: The boundary condition type as a BoundaryTypes object
        :param positions: The node position(s) of this boundary condition as a ixj array
        :param magnitude_function: Magnitude(s) of the bc. Can be either float or function of time f(t)
        :param location: Specify the location of the bc. Of type BoundaryLocations
        :param neumann_direction: if type is Neumann, specify whether in x (0) or y (1) direction
        :param particle_interaction: Action to perform upon particle crossing this bc. Of type BoundaryParticleInteraction
        """
        self.location = location
        self.particle_interaction = particle_interaction
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

        # check whether the magnitude is a function of time, or just static
        if callable(magnitude_function):
            self.magnitude_function = magnitude_function
            self.values = magnitude_function(t=0)
            self.dynamic = True
        else:
            self.values = magnitude_function
            self.dynamic = False

        if location == BoundaryLocations.INTERIOR:
            self.left_node_x = min(positions[0])
            self.right_node_x = max(positions[0])
            self.lower_node_y = min(positions[1])
            self.upper_node_y = max(positions[1])

    def update(self, t):
        """This function updates the magnitude of this boundary condition."""
        self.values = self.magnitude_function(t=t)

    def apply_particle_interaction(self, particle_positions, particle_velocities, x_length, y_length, delta_x, delta_y):
        # TODO: Docstring
        # TODO: Should boundaries gather charge?
        # TODO: Absorbtion emission interactions, secondary electrons
        num_particles = particle_positions.shape[1]
        particles_to_delete = []

        # if DESTROY then we append those particle indices to the list of particles to delete
        # if REFLECT then we reflect their position across the boundary axis back into the system, and we
        # reverse the correct component of velocity.
        if self.location == BoundaryLocations.RIGHT:
            right_mask = particle_positions[0] >= x_length  # mask selects all particles that have passed boundary
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[right_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_positions[0][right_mask] = x_length - (particle_positions[0][right_mask] - x_length)  # reflect back into system
                particle_velocities[0][right_mask] = -particle_velocities[0][right_mask]  # invert velocity

        elif self.location == BoundaryLocations.LEFT:
            left_mask = particle_positions[0] < 0
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[left_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_positions[0][left_mask] = -particle_positions[0][left_mask]
                particle_velocities[0][left_mask] = -particle_velocities[0][left_mask]

        elif self.location == BoundaryLocations.UPPER:
            upper_mask = particle_positions[1] >= y_length
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[upper_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_positions[1][upper_mask] = y_length - (particle_positions[1][upper_mask] - y_length)
                particle_velocities[1][upper_mask] = -particle_velocities[1][upper_mask]

        elif self.location == BoundaryLocations.LOWER:
            lower_mask = particle_positions[1] < 0
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[lower_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_positions[1][lower_mask] = -particle_positions[1][lower_mask]
                particle_velocities[1][lower_mask] = -particle_velocities[1][lower_mask]

        elif self.location == BoundaryLocations.INTERIOR:
            interior_mask = (particle_positions[0] >= self.left_node_x*delta_x) & \
                            (particle_positions[0] < self.right_node_x*delta_x) & \
                            (particle_positions[1] >= self.lower_node_y*delta_y) & \
                            (particle_positions[1] < self.upper_node_y*delta_y)  # check if inside the box
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[interior_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                raise ValueError("Code cannot handle interior reflections :(")

        return particle_positions, particle_velocities, particles_to_delete