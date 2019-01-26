import numpy as np
import time
from BoundaryClasses import BoundaryParticleInteraction
from PotentialSolver import PotentialSolver
from Integrator import LeapfrogIntegrator
import scipy.interpolate

class ParticleManager:
    """Handles all particle / grid operations. Main simulation class."""
    def __init__(self, particle_positions, particle_velocities, particle_charges, particle_masses,
                delta_t, eps_0,
                num_x_nodes, num_y_nodes, delta_x, delta_y,
                boundary_conditions=[],
                particle_sources=[],
                left_boundary_particle_interaction=BoundaryParticleInteraction.OPEN,
                right_boundary_particle_interaction=BoundaryParticleInteraction.OPEN,
                upper_boundary_particle_interaction=BoundaryParticleInteraction.OPEN,
                lower_boundary_particle_interaction=BoundaryParticleInteraction.OPEN,
                integration_method="LEAPFROG"):
        """
        D = number of dimensions (x = 0, y = 1)
        N = number of particles currently in system
        X = number of nodes in x direction
        Y = number of nodes in y direction
        :param particle_positions: DxN array of particle positions
        :param particle_velocities: DxN array of particle velocities
        :param particle_charges: 1xN array of particle charges
        :param particle_masses: 1xN array of particle masses

        :param delta_t: size of time step
        :param eps_0: permitivitty of free space

        :param num_x_nodes: number of nodes in x direction
        :param num_y_nodes: number of nodes in y direction
        :param delta_x: side length of grid cells in x direction
        :param delta_y: side length of grid cells in y direction

        :param boundary_conditions: list of BoundaryCondition objects

        :param particle_sources: list of ParticleSource objects used to generate particle during simulation

        :param left_boundary_particle_interaction: BoundaryParticleInteraction object corresponding with left boundary
        :param right_boundary_particle_interaction: BoundaryParticleInteraction object corresponding with right boundary
        :param upper_boundary_particle_interaction: BoundaryParticleInteraction object corresponding with upper boundary
        :param lower_boundary_particle_interaction: BoundaryParticleInteraction object corresponding with lower boundary

        :param integration_method: type of integration method to use. Options:{LEAPFROG}
        """
        self.start_time = time.time()

        self.dimensions = 2  # This is a 2D PIC simulation, must remain 2 for now.

        self.dt = delta_t
        self.current_t_step = 0  # keeps track of how many timesteps in we are

        self.num_particles = particle_positions.shape[1]  # the number of particles currently in system

        # the following arrays store properties of particles
        self.particle_positions = particle_positions
        self.floored_positions = np.zeros((self.dimensions, self.num_particles))
        self.particle_velocities = particle_velocities
        self.particle_masses = particle_masses
        self.particle_charges = particle_charges
        self.particle_E = np.zeros((self.dimensions, self.num_particles))
        self.particle_forces = np.zeros((self.dimensions, self.num_particles))

        # the following arrays store grid parameters
        self.num_x_nodes = num_x_nodes
        self.num_y_nodes = num_y_nodes
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.x_length = (self.num_x_nodes-1)*self.delta_x  # the total length of the system in x direction
        self.y_length = (self.num_y_nodes-1) * self.delta_y  # the total length of the system in y direction

        # store particle node weightings
        self.w1 = np.zeros(self.num_particles)
        self.w2 = np.zeros(self.num_particles)
        self.w3 = np.zeros(self.num_particles)
        self.w4 = np.zeros(self.num_particles)

        # store the boundary conditions
        self.left_boundary_particle_interaction = left_boundary_particle_interaction
        self.right_boundary_particle_interaction = right_boundary_particle_interaction
        self.upper_boundary_particle_interaction = upper_boundary_particle_interaction
        self.lower_boundary_particle_interaction = lower_boundary_particle_interaction

        self.boundary_conditions = boundary_conditions

        # the following arrays store current properties of the grid nodes
        self.grid_charge_densities = np.zeros((self.num_x_nodes, self.num_y_nodes))
        self.grid_potentials = np.zeros((self.num_x_nodes, self.num_y_nodes))
        self.grid_E = np.zeros((self.num_x_nodes, self.num_y_nodes, self.dimensions))

        self.particle_sources = particle_sources  # list of all particle sources

        self.potential_solver = PotentialSolver(num_x_nodes, num_y_nodes, delta_x, delta_y, eps_0, boundary_conditions)

        if integration_method == "LEAPFROG":
            self.integrator = LeapfrogIntegrator()

            # push back initial velocities back half a time step
            self.calculate_particle_node_weights()
            self.solve_grid_charge_densities()
            self.solve_grid_potentials()
            self.solve_grid_E()
            self.solve_particle_E()
            self.solve_particle_forces()
            self.particle_velocities = self.integrator.push_back_velocity_half_step(particle_velocities,
                                                                                self.particle_forces,
                                                                                self.particle_masses, self.dt)

    def calculate_particle_node_weights(self):
        """Calculates the weighting for each particle for each of the four nodes it is closest to. The weighting
        is equivalent to the area of the square mode in the opposite corner of the node in question. The weighting
        is used to distribute charge from the particles to the nodes and gather the E field felt by the particle from
        the nodes."""
        cell_length_distance = self.particle_positions / [[self.delta_x], [self.delta_y]]
        self.floored_positions = np.floor(cell_length_distance).astype(int)
        fractional_distance = (cell_length_distance - self.floored_positions)

        # these weights are based on how close the particle is to that node.
        self.w1 = (1 - fractional_distance[0, :]) * (1 - fractional_distance[1, :])  # bottom left
        self.w2 = (fractional_distance[0, :]) * (1 - fractional_distance[1, :])  # bottom right
        self.w3 = (1 - fractional_distance[0, :]) * (fractional_distance[1, :])  # top left
        self.w4 = (fractional_distance[0, :]) * (fractional_distance[1, :])  # top right

    def solve_grid_charge_densities(self):
        """Updates the charge density at each node point. Distributes the charge of each particle between the
        four closest nodes, weighted by distance to that node."""
        # setup array to store densities
        self.grid_charge_densities = np.zeros((self.num_x_nodes, self.num_y_nodes))

        # distribute weighted charges of each particle to corresponding node corners
        self.grid_charge_densities[self.floored_positions[0], self.floored_positions[1]] += self.w1 * self.particle_charges
        self.grid_charge_densities[self.floored_positions[0]+1, self.floored_positions[1]] += self.w2 * self.particle_charges
        self.grid_charge_densities[self.floored_positions[0], self.floored_positions[1]+1] += self.w3 * self.particle_charges
        self.grid_charge_densities[self.floored_positions[0]+1, self.floored_positions[1]+1] += self.w4 * self.particle_charges

        self.grid_charge_densities /= (self.delta_x * self.delta_y)  # divide by the cell volume to get density

    def solve_grid_potentials(self):
        """Updates the potentials on the grid based on charge densities"""
        self.grid_potentials = self.potential_solver.solve_potentials(self.grid_charge_densities)

    def solve_grid_E(self):
        """Updates the electric field at each node"""
        # calculate E from central differencing
        # E = d(potential)/dx = -(potential(i+1)-potential(i-1))/2dx
        self.grid_E = np.zeros((self.num_x_nodes, self.num_y_nodes, 2))

        # for boundaries, we need to use forward / backward differences
        self.grid_E[1:-1, :, 0] = (self.grid_potentials[0:-2, :] - self.grid_potentials[2:, :]) / 2 / self.delta_x
        self.grid_E[:, 1:-1, 1] = (self.grid_potentials[:, 0:-2] - self.grid_potentials[:, 2:]) / 2 / self.delta_y
        self.grid_E[0, :, 0] = (self.grid_potentials[0, :] - self.grid_potentials[1, :]) / self.delta_x
        self.grid_E[-1, :, 0] = (self.grid_potentials[-2, :] - self.grid_potentials[-1, :]) / self.delta_x
        self.grid_E[:, 0, 1] = (self.grid_potentials[:, 0] - self.grid_potentials[:, 1]) / self.delta_y
        self.grid_E[:, -1, 1] = (self.grid_potentials[:, -2] - self.grid_potentials[:, -1]) / self.delta_y

    def solve_particle_E(self):
        """Updates the electric field "felt" by each particle. Bilinearly interpolates grid E values onto particle
        positions."""
        self.particle_E = (self.grid_E[self.floored_positions[0], self.floored_positions[1], :].T * self.w1 +
                           self.grid_E[self.floored_positions[0]+1, self.floored_positions[1], :].T * self.w2 +
                           self.grid_E[self.floored_positions[0], self.floored_positions[1]+1, :].T * self.w3 +
                           self.grid_E[self.floored_positions[0]+1, self.floored_positions[1]+1, :].T * self.w4)

    def solve_particle_forces(self):
        """Updates the force felt by each particle."""
        # F = Eq
        self.particle_forces = self.particle_E * self.particle_charges

    def apply_boundary_conditions(self):
        """Checks if particles have left the system and applies the appropriate boundary condition."""
        particles_to_delete = []  # this list will store all particle indices that correspond to out of bounds particles

        # masks to determine all particles that are outside the boundaries
        right_mask = self.particle_positions[0] >= self.x_length
        left_mask = self.particle_positions[0] < 0
        upper_mask = self.particle_positions[1] >= self.y_length
        lower_mask = self.particle_positions[1] < 0

        # if OPEN then we append those particle indices to the list of particles to delete
        # if REFLECTIVE then we reflect their position across the boundary axis back into the system, and we
        # reverse the correct component of velocity.
        if self.right_boundary_particle_interaction == BoundaryParticleInteraction.OPEN:
            particles_to_delete.append(np.arange(self.num_particles)[right_mask])
        elif self.right_boundary_particle_interaction == BoundaryParticleInteraction.REFLECTIVE:
            self.particle_positions[0][right_mask] = self.x_length - (self.particle_positions[0][right_mask] - self.x_length)
            self.particle_velocities[0][right_mask] = -self.particle_velocities[0][right_mask]

        if self.left_boundary_particle_interaction == BoundaryParticleInteraction.OPEN:
            particles_to_delete.append(np.arange(self.num_particles)[left_mask])
        elif self.left_boundary_particle_interaction == BoundaryParticleInteraction.REFLECTIVE:
            self.particle_positions[0][left_mask] = -self.particle_positions[0][left_mask]
            self.particle_velocities[0][left_mask] = -self.particle_velocities[0][left_mask]

        if self.upper_boundary_particle_interaction == BoundaryParticleInteraction.OPEN:
            particles_to_delete.append(np.arange(self.num_particles)[upper_mask])
        elif self.upper_boundary_particle_interaction == BoundaryParticleInteraction.REFLECTIVE:
            self.particle_positions[1][upper_mask] = self.y_length - (self.particle_positions[1][upper_mask] - self.y_length)
            self.particle_velocities[1][upper_mask] = -self.particle_velocities[1][upper_mask]

        if self.lower_boundary_particle_interaction == BoundaryParticleInteraction.OPEN:
            particles_to_delete.append(np.arange(self.num_particles)[lower_mask])
        elif self.lower_boundary_particle_interaction == BoundaryParticleInteraction.REFLECTIVE:
            self.particle_positions[1][lower_mask] = -self.particle_positions[1][lower_mask]
            self.particle_velocities[1][lower_mask] = -self.particle_velocities[1][lower_mask]

        # delete all particles that have left the system
        if len(particles_to_delete) > 0:
            self.delete_particles(particles_to_delete)

    def delete_particles(self, particles_to_delete):
        """
        Deletes particles from the system
        :param particles_to_delete: list of particle indices that are to be deleted
        """
        self.particle_positions = np.delete(self.particle_positions, particles_to_delete, 1)
        self.particle_velocities = np.delete(self.particle_velocities, particles_to_delete, 1)
        self.particle_masses = np.delete(self.particle_masses, particles_to_delete)
        self.particle_charges = np.delete(self.particle_charges, particles_to_delete)
        self.particle_forces = np.delete(self.particle_forces, particles_to_delete, 1)
        self.particle_E = np.delete(self.particle_E, particles_to_delete, 1)
        self.num_particles -= len(particles_to_delete)

    def add_particles(self, particle_source):
        """
        Adds particles to the system
        :param particle_source: ParticleSource object
        """
        num_new_particles = particle_source.new_particle_positions.shape[1]

        self.particle_positions = np.append(self.particle_positions, particle_source.new_particle_positions, 1)
        self.particle_velocities = np.append(self.particle_velocities, particle_source.new_particle_velocities, 1)
        self.particle_masses = np.append(self.particle_masses, particle_source.new_particle_masses)
        self.particle_charges = np.append(self.particle_charges, particle_source.new_particle_charges)
        self.particle_forces = np.append(self.particle_forces, np.zeros((2, num_new_particles)), 1)
        self.particle_E = np.append(self.particle_E, np.zeros((2, num_new_particles)), 1)

        self.num_particles += num_new_particles

    def update_step(self):
        """Updates the simulation forward one time step"""

        # add particles from sources
        for particle_source in self.particle_sources:
            if (self.current_t_step*self.dt) % (1/particle_source.frequency) == 0:
                self.add_particles(particle_source)

        # update dynamic boundary conditions
        for boundary_condition in self.boundary_conditions:
            if boundary_condition.dynamic_values_function is not None:
                boundary_condition.update(self.current_t_step * self.dt)

        # run methods in the correct order
        self.calculate_particle_node_weights()
        self.solve_grid_charge_densities()
        self.solve_grid_potentials()
        self.solve_grid_E()
        self.solve_particle_E()
        self.solve_particle_forces()
        self.particle_positions, self.particle_velocities = self.integrator.integrate(self.particle_positions,
                                                                                      self.particle_velocities,
                                                                                      self.particle_forces,
                                                                                      self.particle_masses, self.dt)
        self.apply_boundary_conditions()

        self.current_t_step += 1

