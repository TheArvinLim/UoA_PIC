import numpy as np


class ParticleSource:
    """Stores information about a particle source in the system"""

    def __init__(self, new_particle_positions, new_particle_velocities, new_particle_masses, new_particle_charges,
                 new_particle_types, frequency=None):
        """
        D = number of dimensions
        N = number of particles
        :param new_particle_positions: DxN array of where the new particles will spawn, or function that returns DxN array
        :param new_particle_velocities: DxN array of the new particles' velocities, or function that returns DxN array
        :param new_particle_masses: 1xN array of the new particles' masses
        :param new_particle_charges: 1xN array of the new particles' charges
        :param frequency: frequency (hz) that this source should make new particles
        """

        if callable(new_particle_positions):
            self.particle_positions_constant = False
            self.particle_positions_func = new_particle_positions
        else:
            self.particle_positions_constant = True
            self.particle_positions = np.array(new_particle_positions)

        if callable(new_particle_velocities):
            self.particle_velocities_constant = False
            self.particle_velocities_func = new_particle_velocities
        else:
            self.particle_velocities_constant = True
            self.particle_velocities = np.array(new_particle_velocities)

        self.particle_masses = np.array(new_particle_masses)
        self.particle_charges = np.array(new_particle_charges)
        self.particle_types = np.array(new_particle_types)
        self.frequency = frequency

    def generate_particle_positions(self):
        if self.particle_positions_constant:
            return self.particle_positions
        else:
            return self.particle_positions_func()

    def generate_particle_velocities(self):
        if self.particle_velocities_constant:
            return self.particle_velocities
        else:
            return self.particle_velocities_func()

    def add_particles(self, particle_system):
        if np.floor(particle_system.simulation_time * particle_system.frequency) > \
                np.floor((particle_system.simulation_time - particle_system.delta_t) * particle_system.frequency):
            particle_system.add_particles(self.generate_particle_positions(),
                                          self.generate_particle_velocities(),
                                          self.particle_masses, self.particle_charges, self.particle_types)



