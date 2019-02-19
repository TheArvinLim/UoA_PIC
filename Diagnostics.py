import numpy as np
import matplotlib.pyplot as plt
from ParticleTypes import*

# TODO: Velocity histogram
# TODO: Export simulation data to file


class ParticleSystemHistory:
    # data structures for saving snapshots
    def __init__(self, particle_system):
        self.particle_system = particle_system

        self.simulation_time_history = []

        self.grid_potentials_history = []
        self.grid_charge_densities_history = []

        self.particle_positions_history = []
        self.particle_velocities_history = []
        self.particle_types_history = []
        self.particle_masses_history = []

        self.grid_E_history = []

    def take_snapshot(self):
        self.simulation_time_history.append(self.particle_system.simulation_time)

        self.grid_potentials_history.append(self.particle_system.grid_potentials)
        self.grid_charge_densities_history.append(self.particle_system.grid_charge_densities)

        self.particle_positions_history.append(self.particle_system.particle_positions)
        self.particle_velocities_history.append(self.particle_system.particle_velocities)
        self.particle_types_history.append(self.particle_system.particle_types)
        self.particle_masses_history.append(self.particle_system.particle_masses)

        self.grid_E_history.append(self.particle_system.grid_E)

    def plot_energy_history(self):
        """Calculates and plots stored energy histories"""
        energy_potential_history = []
        energy_kinetic_history = []
        energy_total_history = []

        for i in range(len(self.particle_positions_history)):
            total_potential = np.sum(1 / 2 * self.grid_potentials_history[i] * self.grid_charge_densities_history[i] *
                                     self.particle_system.delta_x * self.particle_system.delta_y)

            total_kinetic = np.sum(1 / 2 * self.particle_masses_history[i] *
                                   np.linalg.norm(self.particle_velocities_history[i].T, axis=1) ** 2)

            total_energy = total_kinetic + total_potential

            energy_potential_history = energy_potential_history + [total_potential]
            energy_kinetic_history = energy_kinetic_history + [total_kinetic]
            energy_total_history = energy_total_history + [total_energy]

        plt.plot(energy_potential_history, label='potential')
        plt.plot(energy_kinetic_history, label='kinetic')
        plt.plot(energy_total_history, label='total')
        plt.legend(loc='center right')
        plt.show()

    def plot_average_potential(self):
        avr_grid_potential = np.zeros(np.shape(self.grid_potentials_history[0]))
        for i in self.grid_potentials_history:
            avr_grid_potential += i
        avr_grid_potential /= len(self.grid_potentials_history)

        plt.imshow(avr_grid_potential.T,
               vmin=np.min(avr_grid_potential),
               vmax=np.max(avr_grid_potential),
               interpolation="bicubic", origin="lower",
               extent=[0, self.particle_system.x_length, 0, self.particle_system.y_length])
        plt.title("Averaged potential")
        plt.show()

        plt.plot(np.mean(avr_grid_potential.T, axis=0))
        plt.title("Averaged potential along x")
        plt.show()

    def plot_average_charge_density(self):
        avr_grid_charge_density = np.zeros(np.shape(self.grid_charge_densities_history[0]))
        for i in self.grid_charge_densities_history:
            avr_grid_charge_density += i
            avr_grid_charge_density /= len(self.grid_charge_densities_history)

        plt.plot(np.mean(avr_grid_charge_density.T, axis=0))
        plt.title("Averaged charge density along x")
        plt.show()

    def plot_particle_number(self):
        electron_count_history = [np.sum(particle_types == ParticleTypes.ELECTRON) for particle_types in self.particle_types_history]
        argon_ion_count_history = [np.sum(particle_types == ParticleTypes.ARGON_ION) for particle_types in self.particle_types_history]

        plt.plot(electron_count_history, label='Electrons')
        plt.plot(argon_ion_count_history, label='Ions')
        plt.show()