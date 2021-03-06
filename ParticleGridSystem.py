import numpy as np
import time
from BoundaryClasses import BoundaryLocations
from PotentialSolver import PotentialSolver
from Integrator import LeapfrogIntegrator

# TODO: Allow for r-z coordinate system with axisymmetric boundary condition to simulate cylindrical geometry
# TODO: Allow for modelling charge density due to electrons via the Boltzmann relation (not necessary?)
# TODO: Convert to 3D
# TODO: Implement electromagnetic effects (in the low-temp, high-pressure, rf-driven discharge limit, magnetic effects
#  are negligible)
# TODO: Normalise units to delta_x = 1, delta_t = 1
# TODO: Electron ion subcycling (different dt for species)
# TODO: Model neutrals as having a Maxwellian velocity distribution and a uniform density within the boundaries


class ParticleGridSystem:
    """Handles all particle / grid operations. Main simulation class."""
    def __init__(self,
                delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length,
                particle_positions=np.array([[], []]),
                particle_velocities=np.array([[], []]),
                particle_charges=np.array([]),
                particle_masses=np.array([]),
                particle_types=np.array([]),
                boundary_conditions=[],
                particle_sources=[],
                integration_method="LEAPFROG",
                collision_scheme="NONE"):
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
        :param eps_0: permittivity of free space

        :param num_x_nodes: number of nodes in x direction
        :param num_y_nodes: number of nodes in y direction

        :param boundary_conditions: list of BoundaryCondition objects. Should set all 4 edges, plus any interior bcs.

        :param particle_sources: list of ParticleSource objects used to generate particle during simulation

        :param integration_method: type of integration method to use. Options:{LEAPFROG}
        :param collision_scheme: collision model. Options: {NONE, MONTE_CARLO}
        """
        self.start_time = time.time()
        self.simulation_time = 0  # keeps track of time within the simulation (s)

        self.dimensions = 2  # This is a 2D PIC simulation, must remain 2 for now.

        self.delta_t = delta_t
        self.current_t_step = 0  # keeps track of how many timesteps in we are

        self.num_particles = particle_positions.shape[1]  # the number of particles currently in system

        # the following arrays store properties of particles
        self.particle_positions = particle_positions
        self.particle_velocities = particle_velocities
        self.particle_masses = particle_masses
        self.particle_charges = particle_charges
        self.particle_E = np.zeros((self.dimensions, self.num_particles))
        self.particle_forces = np.zeros((self.dimensions, self.num_particles))
        self.particle_types = particle_types

        # the following arrays store grid parameters
        self.num_x_nodes = num_x_nodes
        self.num_y_nodes = num_y_nodes
        self.delta_x = x_length / (self.num_x_nodes-1)
        self.delta_y = y_length / (self.num_y_nodes-1)
        self.x_length = x_length
        self.y_length = y_length

        # set boundary conditions and boundary particle interactions
        self.boundary_conditions = boundary_conditions
        bc_locations = []
        for boundary_condition in boundary_conditions:
            bc_locations.append(boundary_condition.location)
        if BoundaryLocations.LEFT not in bc_locations:
            raise ValueError("Left boundary particle interaction has not been specified.")
        if BoundaryLocations.RIGHT not in bc_locations:
            raise ValueError("Right boundary particle interaction has not been specified.")
        if BoundaryLocations.UPPER not in bc_locations:
            raise ValueError("Lower boundary particle interaction has not been specified.")
        if BoundaryLocations.LOWER not in bc_locations:
            raise ValueError("Upper boundary particle interaction has not been specified.")

        # the following arrays store current properties of the grid nodes
        self.grid_charge_densities = np.zeros((self.num_x_nodes, self.num_y_nodes))
        self.grid_potentials = np.zeros((self.num_x_nodes, self.num_y_nodes))
        self.grid_E = np.zeros((self.num_x_nodes, self.num_y_nodes, self.dimensions))

        self.particle_sources = particle_sources  # list of all particle sources

        self.potential_solver = PotentialSolver(num_x_nodes, num_y_nodes, self.delta_x, self.delta_y, eps_0, boundary_conditions)

        if integration_method == "LEAPFROG":
            self.integrator = LeapfrogIntegrator()

            # push back initial velocities back half a time step
            self.solve_particle_forces()
            self.particle_velocities = self.integrator.push_back_velocity_half_step(particle_velocities,
                                                                                self.particle_forces,
                                                                                self.particle_masses, self.delta_t)

        self.collision_scheme = collision_scheme

    def solve_particle_forces(self):
        # Calculates the weighting for each particle for each of the four nodes it is closest to. The weighting
        # is equivalent to the area of the square mode in the opposite corner of the node in question. The weighting
        # is used to distribute charge from the particles to the nodes and gather the E field felt by the particle from
        # the nodes. This is a first order (linear) weighting scheme.

        # TODO: Implement options for higher order schemes

        cell_length_distance = self.particle_positions / [[self.delta_x], [self.delta_y]]
        floored_positions = np.floor(cell_length_distance).astype(int)
        fractional_distance = (cell_length_distance - floored_positions)

        # these weights are based on how close the particle is to that node.
        w1 = (1 - fractional_distance[0]) * (1 - fractional_distance[1])  # bottom left
        w2 = (fractional_distance[0]) * (1 - fractional_distance[1])  # bottom right
        w3 = (1 - fractional_distance[0]) * (fractional_distance[1])  # top left
        w4 = (fractional_distance[0]) * (fractional_distance[1])  # top right

        # Updates the charge density at each node point. Distributes the charge of each particle between the
        # four closest nodes, weighted by distance to that node.
        # setup array to store densities
        self.grid_charge_densities = np.zeros((self.num_x_nodes, self.num_y_nodes))

        # we need to use np.add.at since it is equivalent to a[indices] += b, except that results are accumulated
        # for elements that are indexed more than once (which occurs if two particles are in the same cell)
        np.add.at(self.grid_charge_densities, (floored_positions[0], floored_positions[1]),
                  w1 * self.particle_charges)
        np.add.at(self.grid_charge_densities, (floored_positions[0]+1, floored_positions[1]),
                  w2 * self.particle_charges)
        np.add.at(self.grid_charge_densities, (floored_positions[0], floored_positions[1]+1),
                  w3 * self.particle_charges)
        np.add.at(self.grid_charge_densities, (floored_positions[0]+1, floored_positions[1]+1),
                  w4 * self.particle_charges)

        # add boundary charges
        for boundary in self.boundary_conditions:
            self.grid_charge_densities[boundary.positions[0], boundary.positions[1]] += boundary.node_charges

        self.grid_charge_densities /= (self.delta_x * self.delta_y)  # divide by the cell volume to get density

        # Updates the potentials on the grid based on charge densities
        self.grid_potentials = self.potential_solver.solve_potentials(self.grid_charge_densities)

        # Updates the electric field at each node
        self.grid_E = np.zeros((self.num_x_nodes, self.num_y_nodes, 2))

        # calculate E from central differencing
        # E = d(potential)/dx = -(potential(i+1)-potential(i-1))/2dx
        self.grid_E[1:-1, :, 0] = (self.grid_potentials[0:-2, :] - self.grid_potentials[2:, :]) / 2 / self.delta_x  # x
        self.grid_E[:, 1:-1, 1] = (self.grid_potentials[:, 0:-2] - self.grid_potentials[:, 2:]) / 2 / self.delta_y  # y

        # for boundaries, we need to use forward / backward differences
        self.grid_E[0, :, 0] = (self.grid_potentials[0, :] - self.grid_potentials[1, :]) / self.delta_x  # left
        self.grid_E[-1, :, 0] = (self.grid_potentials[-2, :] - self.grid_potentials[-1, :]) / self.delta_x  # right
        self.grid_E[:, 0, 1] = (self.grid_potentials[:, 0] - self.grid_potentials[:, 1]) / self.delta_y  # lower
        self.grid_E[:, -1, 1] = (self.grid_potentials[:, -2] - self.grid_potentials[:, -1]) / self.delta_y  # upper

        # Updates the electric field "felt" by each particle. Bi-linearly interpolates grid E values onto particle
        # positions.
        self.particle_E = (self.grid_E[floored_positions[0], floored_positions[1]].T * w1 +
                           self.grid_E[floored_positions[0] + 1, floored_positions[1]].T * w2 +
                           self.grid_E[floored_positions[0], floored_positions[1] + 1].T * w3 +
                           self.grid_E[floored_positions[0] + 1, floored_positions[1] + 1].T * w4)

        # Updates the force felt by each particle.
        # F = Eq
        self.particle_forces = self.particle_E * self.particle_charges

    def resolve_collisions(self):
        # TODO: Implement Monte Carlo scheme
        pass
        # P = 1 - exp(-v*dt/(mean free path))
        # if self.collision_scheme == "MONTE_CARLO":
        #     for particle_velocity in self.particle_velocities.T:
        #         particle_velocity_norm = np.linalg.norm(particle_velocity)
        #         #p_collide = 1 - np.exp(-particle_velocity_norm*self.dt/self.mean_free_path())

    def delete_particles(self, particles_to_delete):
        """
        Deletes particles from the system
        :param particles_to_delete: list of particle indices that are to be deleted
        """
        # there is a possibility that duplicate indices are passed into this function, we only delete the unique
        # indices.
        particles_to_delete = np.unique(particles_to_delete).astype(int)

        self.particle_positions = np.delete(self.particle_positions, particles_to_delete, 1)
        self.particle_velocities = np.delete(self.particle_velocities, particles_to_delete, 1)
        self.particle_masses = np.delete(self.particle_masses, particles_to_delete)
        self.particle_charges = np.delete(self.particle_charges, particles_to_delete)
        self.particle_types = np.delete(self.particle_types, particles_to_delete)
        self.particle_forces = np.delete(self.particle_forces, particles_to_delete, 1)
        self.particle_E = np.delete(self.particle_E, particles_to_delete, 1)

        self.num_particles = len(self.particle_masses)

    def add_particles(self, positions, velocities, masses, charges, types):
        """
        Adds particles to the system
        :param positions: array of positions to add particles to
        :param velocities: array of new particle positions
        :param masses: array of new particle masses
        :param charges: array of new particle charges
        :param types: array of new particle types
        """
        num_new_particles = positions.shape[1]

        self.particle_positions = np.append(self.particle_positions, positions, 1)
        self.particle_velocities = np.append(self.particle_velocities, velocities, 1)
        self.particle_masses = np.append(self.particle_masses, masses)
        self.particle_charges = np.append(self.particle_charges, charges)
        self.particle_types = np.append(self.particle_types, types)
        self.particle_forces = np.append(self.particle_forces, np.zeros((2, num_new_particles)), 1)
        self.particle_E = np.append(self.particle_E, np.zeros((2, num_new_particles)), 1)

        self.num_particles += num_new_particles

    def update_step(self):
        """Updates the simulation forward one time step"""
        # update dynamic boundary conditions
        for boundary_condition in self.boundary_conditions:
            boundary_condition.update(self.simulation_time)

        self.solve_particle_forces()

        # integrate equations of motion
        self.particle_positions, self.particle_velocities = self.integrator.integrate(self.particle_positions,
                                                                                      self.particle_velocities,
                                                                                      self.particle_forces,
                                                                                      self.particle_masses, self.delta_t)

        for boundary_condition in self.boundary_conditions:
            boundary_condition.apply_particle_interaction(self)

        self.resolve_collisions()

        # add particles from sources
        for particle_source in self.particle_sources:
            particle_source.add_particles(self)

        self.current_t_step += 1
        self.simulation_time += self.delta_t
