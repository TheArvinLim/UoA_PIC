import numpy as np

# TODO: Give option to initialise new particle velocities from a distribution function.
class ParticleSource:
    """Stores information about a particle source in the system"""

    def __init__(self, new_particle_positions, new_particle_velocities, new_particle_masses, new_particle_charges,
                 frequency):
        """
        D = number of dimensions
        N = number of particles
        :param new_particle_positions: DxN array of where the new particles will spawn
        :param new_particle_velocities: DxN array of the new particles' velocities
        :param new_particle_masses: 1xN array of the new particles' masses
        :param new_particle_charges: 1xN array of the new particles' charges
        :param frequency: frequency (hz) that this source should make new particles
        """
        self.new_particle_positions = np.array(new_particle_positions)
        self.new_particle_velocities = np.array(new_particle_velocities)
        self.new_particle_masses = np.array(new_particle_masses)
        self.new_particle_charges = np.array(new_particle_charges)
        self.frequency = frequency