import numpy as np
import matplotlib.pyplot as plt

# TODO: Velocity histogram
# TODO: Export simulation data to file


class EnergyDiagnostic:
    """Handles all energy related functions."""
    def __init__(self):
        # these store all previously calculated energies
        self.energy_potential_history = []
        self.energy_kinetic_history = []
        self.energy_total_history = []

    def calc_system_energy(self, particle_system):
        """
        Calculates the potential, kinetic and total energies and stores them.
        D = number of dimensions
        N = number of particles
        X = number of nodes in x direction
        Y = number of nodes in y direciton
        :param particle_velocities: DxN array of velocities of particles
        :param particle_masses: 1xN array of masses of particles
        :param grid_potentials: XxY array of node potentials
        :param grid_charge_densities: XxY array of node charge densities
        :param grid_cell_volume: volume of grid cell
        """
        particle_potentials = (1/2 * particle_system.grid_potentials * particle_system.grid_charge_densities *
                               particle_system.delta_x * particle_system.delta_y)

        total_potential = np.sum(particle_potentials)

        total_kinetic = 0
        for i, velocity in enumerate(particle_system.particle_velocities.T):
            total_kinetic += 1/2 * particle_system.particle_masses[i] * np.linalg.norm(velocity) ** 2

        total_energy = total_potential + total_kinetic

        self.energy_potential_history = self.energy_potential_history + [total_potential]
        self.energy_kinetic_history = self.energy_kinetic_history + [total_kinetic]
        self.energy_total_history = self.energy_total_history + [total_energy]

    def plot_energy_history(self):
        """Plots stored energy histories"""
        plt.plot(self.energy_potential_history, label='potential')
        plt.plot(self.energy_kinetic_history, label='kinetic')
        plt.plot(self.energy_total_history, label='total')
        plt.legend(loc='center right')
        plt.show()


class ParticleSystemHistory:
    # data structures for saving snapshots
    def __init__(self):
        self.simulation_time_history = []
        self.grid_potentials_history = []
        self.grid_charge_densities_history = []
        self.particle_positions_history = []
        self.grid_E_history = []

    def take_snapshot(self, particle_system):
        self.simulation_time_history.append(particle_system.simulation_time)
        self.grid_potentials_history.append(particle_system.grid_potentials)
        self.grid_charge_densities_history.append(particle_system.grid_charge_densities)
        self.particle_positions_history.append(particle_system.particle_positions)
        self.grid_E_history.append(particle_system.grid_E)