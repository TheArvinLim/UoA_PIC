import numpy as np


class ParticleSource:
    """Stores information about a particle source in the system"""

    def __init__(self, new_particle_positions, new_particle_velocities, new_particle_masses, new_particle_charges,
                 frequency):
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



