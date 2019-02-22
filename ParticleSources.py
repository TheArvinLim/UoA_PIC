import numpy as np
from PlasmaPhysicsFunctions import*


class ParticleSource:
    """Parent class for all particle sources"""
    def __init__(self):
        pass

    def add_particles(self):
        pass


class StaticParticleEmitter(ParticleSource):
    def __init__(self, new_particle_positions, new_particle_velocities, new_particle_masses, new_particle_charges,
                 new_particle_types, frequency):
        """
        Particle emitter that creates new particles at given position with given velocities at a given frequency
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
        self.particle_masses = np.array(new_particle_masses)
        self.particle_charges = np.array(new_particle_charges)
        self.particle_types = np.array(new_particle_types)
        self.frequency = frequency

    def add_particles(self, particle_system):
        if np.floor(particle_system.simulation_time * particle_system.frequency) > \
                np.floor((particle_system.simulation_time - particle_system.delta_t) * particle_system.frequency):
            particle_system.add_particles(self.new_particle_positions,
                                          self.new_particle_velocities,
                                          self.particle_masses, self.particle_charges, self.particle_types)


class MaintainChargeDensity(ParticleSource):
    #TODO: docstring
    def __init__(self, initial_particle_types, electron_v_thermal, ion_v_thermal, specific_weight):
        self.initialElectronCount = np.sum(initial_particle_types == ParticleTypes.ELECTRON)
        self.initialIonCount = np.sum(initial_particle_types == ParticleTypes.ARGON_ION)
        self.electron_v_thermal = electron_v_thermal
        self.ion_v_thermal = ion_v_thermal
        self.specific_weight = specific_weight

    def add_particles(self, particle_system):
        electrons_to_add = self.initialElectronCount - np.sum(particle_system.particle_types == ParticleTypes.ELECTRON)
        ions_to_add = self.initialIonCount - np.sum(particle_system.particle_types == ParticleTypes.ARGON_ION)

        if electrons_to_add < 0:
            electrons_to_add = 0
        if ions_to_add < 0:
            ions_to_add = 0

        particle_positions = np.random.rand(2, electrons_to_add + ions_to_add) * \
                             [[particle_system.x_length], [particle_system.y_length]]
        # add somewhere random within system

        ion_velocities = sample_maxwell_boltzmann_velocity_distribution(self.ion_v_thermal, ions_to_add)
        electron_velocities = sample_maxwell_boltzmann_velocity_distribution(self.electron_v_thermal, electrons_to_add)
        particle_velocities = np.concatenate((ion_velocities, electron_velocities), axis=1)

        particle_charges = np.append(np.zeros(ions_to_add) + ArgonIon().charge * self.specific_weight,
                                     np.zeros(electrons_to_add) + Electron().charge * self.specific_weight)
        particle_masses = np.append(np.zeros(ions_to_add) + ArgonIon().mass * self.specific_weight,
                                    np.zeros(electrons_to_add) + Electron().mass * self.specific_weight)
        particle_types = np.append([ArgonIon().type] * ions_to_add,
                                   [Electron().type] * electrons_to_add)

        particle_system.add_particles(particle_positions, particle_velocities, particle_masses, particle_charges,
                                      particle_types)




