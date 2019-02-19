import numpy as np
from enum import Enum


class FieldBoundaryCondition(Enum):
    """Stores all possible boundary condition types"""
    NEUMANN = 0  # sets value at boundary to constant
    DIRICHLET = 1  # sets derivative at boundary to constant
    FLOATING = 2


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
    def __init__(self, type, positions, magnitude_function, location, neumann_direction=None, particle_interaction=BoundaryParticleInteraction.DESTROY, collect_charge=False):
        """
        :param type: The boundary condition type as a BoundaryTypes object
        :param positions: The node position(s) of this boundary condition as a ixj array
        :param magnitude_function: Magnitude(s) of the bc. Can be either float or function of time f(t)
        :param location: Specify the location of the bc. Of type BoundaryLocations
        :param neumann_direction: if type is Neumann, specify whether in x (0) or y (1) direction
        :param particle_interaction: Action to perform upon particle crossing this bc. Of type BoundaryParticleInteraction
        :param collect_charge: Boolean of whether to collect charge onto this plate
        """
        self.location = location
        self.particle_interaction = particle_interaction
        self.positions = positions

        min_x = np.min(self.positions[0])
        max_x = np.max(self.positions[0])
        min_y = np.min(self.positions[1])
        max_y = np.max(self.positions[1])
        self.num_x_cells = max_x-min_x + 1
        self.num_y_cells = max_y-min_y + 1

        self.interior_row_mask = (self.positions[0] > min_x) & (self.positions[0] < max_x)
        self.interior_col_mask = (self.positions[1] > min_y) & (self.positions[1] < max_y)
        self.interior_mask = self.interior_row_mask & self.interior_col_mask

        self.type = type
        self.collect_charge = collect_charge
        self.charge = 0
        self.node_charges = np.zeros(len(self.positions[0]))
        self.calc_node_charges()

        # check for direction of the neumann boundary
        if self.type == FieldBoundaryCondition.NEUMANN:
            if neumann_direction is None:
                raise ValueError('Please specify direction of Neumann boundary condition')
            elif not (neumann_direction == 0 or neumann_direction == 1):
                raise ValueError('Please specify Neumann boundary condition as either in '
                                 'x direction (0) or y direction (1)')

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
        if self.dynamic:
            self.values = self.magnitude_function(t=t)

    def apply_particle_interaction(self, particle_system):
        # TODO: Docstring
        # TODO: Absorbtion emission interactions, secondary electrons
        num_particles = particle_system.particle_positions.shape[1]
        particles_to_delete = []

        # if DESTROY then we append those particle indices to the list of particles to delete
        # if REFLECT then we reflect their position across the boundary axis back into the system, and we
        # reverse the correct component of velocity.
        if self.location == BoundaryLocations.RIGHT:
            right_mask = particle_system.particle_positions[0] >= particle_system.x_length  # mask selects all particles that have passed boundary
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[right_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_system.particle_positions[0][right_mask] = particle_system.x_length - (particle_system.particle_positions[0][right_mask] - particle_system.x_length)  # reflect back into system
                particle_system.particle_velocities[0][right_mask] = -particle_system.particle_velocities[0][right_mask]  # invert velocity

            if self.collect_charge:
                self.charge += np.sum(particle_system.particle_charges[right_mask])

        elif self.location == BoundaryLocations.LEFT:
            left_mask = particle_system.particle_positions[0] < 0
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[left_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_system.particle_positions[0][left_mask] = -particle_system.particle_positions[0][left_mask]
                particle_system.particle_velocities[0][left_mask] = -particle_system.particle_velocities[0][left_mask]

            if self.collect_charge:
                self.charge += np.sum(particle_system.particle_charges[left_mask])

        elif self.location == BoundaryLocations.UPPER:
            upper_mask = particle_system.particle_positions[1] >= particle_system.y_length
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[upper_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_system.particle_positions[1][upper_mask] = particle_system.y_length - (particle_system.particle_positions[1][upper_mask] - particle_system.y_length)
                particle_system.particle_velocities[1][upper_mask] = -particle_system.particle_velocities[1][upper_mask]

            if self.collect_charge:
                self.charge += np.sum(particle_system.particle_charges[upper_mask])

        elif self.location == BoundaryLocations.LOWER:
            lower_mask = particle_system.particle_positions[1] < 0
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[lower_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                particle_system.particle_positions[1][lower_mask] = -particle_system.particle_positions[1][lower_mask]
                particle_system.particle_velocities[1][lower_mask] = -particle_system.particle_velocities[1][lower_mask]

            if self.collect_charge:
                self.charge += np.sum(particle_system.particle_charges[lower_mask])

        elif self.location == BoundaryLocations.INTERIOR:
            interior_mask = (particle_system.particle_positions[0] >= self.left_node_x*particle_system.delta_x) & \
                            (particle_system.particle_positions[0] < self.right_node_x*particle_system.delta_x) & \
                            (particle_system.particle_positions[1] >= self.lower_node_y*particle_system.delta_y) & \
                            (particle_system.particle_positions[1] < self.upper_node_y*particle_system.delta_y)  # check if inside the box
            if self.particle_interaction == BoundaryParticleInteraction.DESTROY:
                particles_to_delete = np.arange(num_particles)[interior_mask]
            elif self.particle_interaction == BoundaryParticleInteraction.REFLECT:
                raise ValueError("Code cannot handle interior reflections :(")

            if self.collect_charge:
                self.charge += np.sum(particle_system.particle_charges[interior_mask])

        # update charge_densities
        if self.collect_charge:
            self.calc_node_charges()

        particle_system.delete_particles(particles_to_delete)

    def calc_node_charges(self):
        charge_per_cell = self.charge / self.num_x_cells / self.num_y_cells

        self.node_charges = np.zeros(len(self.positions[0]))
        self.node_charges += charge_per_cell
        self.node_charges[self.interior_row_mask] += charge_per_cell
        self.node_charges[self.interior_col_mask] += charge_per_cell
        self.node_charges[self.interior_mask] += charge_per_cell


class InteriorPlate(BoundaryCondition):
    def __init__(self, lower_left_corner, upper_right_corner, value, bc_type, interaction, collect_charge=False, neumann_direction=None):
        xsize = upper_right_corner[0] - lower_left_corner[0] + 1
        ysize = upper_right_corner[1] - lower_left_corner[1] + 1

        xs = np.zeros((xsize * ysize))
        ys = np.zeros((xsize * ysize))

        for i in range(xsize):
            for j in range(ysize):
                xs[j + i * ysize] = i + lower_left_corner[0]
                ys[j + i * ysize] = j + lower_left_corner[1]

        super().__init__(bc_type, np.array([xs.astype(int), ys.astype(int)]), value, BoundaryLocations.INTERIOR,
                         neumann_direction=neumann_direction, particle_interaction=interaction,
                         collect_charge=collect_charge)


class LeftBoundary(BoundaryCondition):
    def __init__(self, num_x_nodes, num_y_nodes, bc_type, value, interaction, collect_charge):
        neumann_direction = None
        xs = np.zeros(num_y_nodes)
        ys = np.arange(num_y_nodes)
        location = BoundaryLocations.LEFT
        neumann_direction = 0

        super().__init__(bc_type, np.array([xs.astype(int), ys.astype(int)]), value, location,
                         neumann_direction=neumann_direction, particle_interaction=interaction,
                         collect_charge=collect_charge)


class RightBoundary(BoundaryCondition):
    def __init__(self, num_x_nodes, num_y_nodes, bc_type, value, interaction, collect_charge):
        neumann_direction = None
        xs = np.zeros(num_y_nodes) + num_x_nodes - 1
        ys = np.arange(num_y_nodes)
        location = BoundaryLocations.RIGHT
        neumann_direction = 0

        super().__init__(bc_type, np.array([xs.astype(int), ys.astype(int)]), value, location,
                         neumann_direction=neumann_direction, particle_interaction=interaction,
                         collect_charge=collect_charge)


class UpperBoundary(BoundaryCondition):
    def __init__(self, num_x_nodes, num_y_nodes, bc_type, value, interaction, collect_charge):
        neumann_direction = None
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes) + num_y_nodes - 1
        location = BoundaryLocations.UPPER
        neumann_direction = 1

        super().__init__(bc_type, np.array([xs.astype(int), ys.astype(int)]), value, location,
                         neumann_direction=neumann_direction, particle_interaction=interaction,
                         collect_charge=collect_charge)


class LowerBoundary(BoundaryCondition):
    def __init__(self, num_x_nodes, num_y_nodes, bc_type, value, interaction, collect_charge):
        neumann_direction = None
        xs = np.arange(num_x_nodes)
        ys = np.zeros(num_x_nodes)
        location = BoundaryLocations.LOWER
        neumann_direction = 1

        super().__init__(bc_type, np.array([xs.astype(int), ys.astype(int)]), value, location,
                         neumann_direction=neumann_direction, particle_interaction=interaction,
                         collect_charge=collect_charge)