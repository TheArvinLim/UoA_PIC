import numpy as np
import matplotlib.pyplot as plt
from PlasmaPhysicsFunctions import*
import numpy as np
import matplotlib.colors as colours
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
# TODO: Velocity histogram
# TODO: docstrings


class SimulationHistorian:
    """Keeps track of the simulation state and saves them"""
    def __init__(self, particle_system):
        self.particle_system = particle_system

        self.simulation_time_history = []

        self.grid_potentials_history = []
        self.grid_charge_densities_history = []

        self.particle_positions_history = []
        self.particle_velocities_history = []
        self.particle_types_history = []
        self.particle_masses_history = []

    def take_snapshot(self):
        self.simulation_time_history.append(self.particle_system.simulation_time)

        self.grid_potentials_history.append(self.particle_system.grid_potentials)
        self.grid_charge_densities_history.append(self.particle_system.grid_charge_densities)

        self.particle_positions_history.append(self.particle_system.particle_positions)
        self.particle_velocities_history.append(self.particle_system.particle_velocities)
        self.particle_types_history.append(self.particle_system.particle_types)
        self.particle_masses_history.append(self.particle_system.particle_masses)

    def save_results(self, filename, simulation_parameters=None):
        """Save simulation data"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dest_dir = os.path.join(script_dir, filename)
        try:
            os.makedirs(dest_dir)
        except OSError:
            pass  # already exists
        path = os.path.join(dest_dir, filename)

        np.savez(path,
                 num_x_nodes=self.particle_system.num_x_nodes,
                 num_y_nodes=self.particle_system.num_y_nodes,
                 delta_x=self.particle_system.delta_x,
                 delta_y=self.particle_system.delta_y,
                 x_length=self.particle_system.x_length,
                 y_length=self.particle_system.y_length,
                 simulation_time_history=self.simulation_time_history,
                 grid_potentials_history=self.grid_potentials_history,
                 grid_charge_densities_history=self.grid_charge_densities_history,
                 particle_positions_history=self.particle_positions_history,
                 particle_velocities_history=self.particle_velocities_history,
                 particle_types_history=self.particle_types_history,
                 particle_masses_history=self.particle_masses_history)

        with open(path + '.txt', 'w') as file:
            file.write(json.dumps(simulation_parameters))  # use `json.loads` to do the reverse

        print("Data successfully saved to", filename + '.npz')

class SimulationDataAnalyser:
    def __init__(self, filename):
        simulation_data = np.load(filename + '.npz')
        self.simulation_time_history = simulation_data['simulation_time_history']
        self.grid_potentials_history = simulation_data['grid_potentials_history']
        self.grid_charge_densities_history = simulation_data['grid_charge_densities_history']
        self.particle_positions_history = simulation_data['particle_positions_history']
        self.particle_velocities_history = simulation_data['particle_velocities_history']
        self.particle_types_history = simulation_data['particle_types_history']
        self.particle_masses_history = simulation_data['particle_masses_history']

        self.num_x_nodes = simulation_data['num_x_nodes']
        self.num_y_nodes = simulation_data['num_y_nodes']
        self.delta_x = simulation_data['delta_x']
        self.delta_y = simulation_data['delta_y']
        self.x_length = simulation_data['x_length']
        self.y_length = simulation_data['y_length']

    def plot_energy_history(self):
        """Calculates and plots stored energy histories"""
        energy_potential_history = []
        energy_kinetic_history = []
        energy_total_history = []

        for i in range(len(self.particle_positions_history)):
            total_potential = np.sum(1 / 2 * self.grid_potentials_history[i] * self.grid_charge_densities_history[i] *
                                     self.delta_x * self.delta_y)

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
                   extent=[0, self.x_length, 0, self.y_length])
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
        electron_count_history = [np.sum(particle_types == ParticleTypes.ELECTRON) for particle_types in
                                  self.particle_types_history]
        argon_ion_count_history = [np.sum(particle_types == ParticleTypes.ARGON_ION) for particle_types in
                                   self.particle_types_history]

        plt.plot(electron_count_history, label='Electrons')
        plt.plot(argon_ion_count_history, label='Ions')
        plt.show()

    def animate_history(self, i):
        # animate particles
        electron_positions = self.particle_positions_history[i].T[
            self.particle_types_history[i] == ParticleTypes.ELECTRON].T
        ion_positions = self.particle_positions_history[i].T[
            self.particle_types_history[i] == ParticleTypes.ARGON_ION].T
        self.electrons.set_data(electron_positions)
        self.ions.set_data(ion_positions)

        # animate charge density
        self.densities.set_data(self.grid_charge_densities_history[i].T)

        # set time text
        self.timetext.set_text(f'{self.simulation_time_history[i] * 1000000:<6.4f}µs')

        # animate potentials
        self.potentials.set_array(self.grid_potentials_history[i].T)

        return self.potentials, self.densities, self.timetext, self.electrons, self.ions

    def begin_animation(self):
        fig = plt.figure(figsize=(6, 8))

        ax3 = fig.add_subplot(311)
        ax3.set_title("Positions")
        ax3.set_xlim(0, self.x_length)
        ax3.set_ylim(0, self.y_length)
        self.ions, = ax3.plot([], [], 'co', markersize=0.5, alpha=0.5)
        self.electrons, = ax3.plot([], [], 'ro', markersize=0.5, alpha=0.5)

        ax1 = fig.add_subplot(312)
        ax1.set_title("Densities")
        self.densities = ax1.imshow(np.zeros((self.num_x_nodes, self.num_y_nodes)),
                               vmin=np.min([np.min(i) for i in self.grid_charge_densities_history]),
                               vmax=np.max([np.max(i) for i in self.grid_charge_densities_history]),
                               interpolation="bicubic", origin="lower",
                               extent=[0, self.x_length, 0, self.y_length])

        ax2 = fig.add_subplot(313)
        ax2.set_title("Potential")
        self.potentials = ax2.imshow(np.zeros((self.num_x_nodes, self.num_y_nodes)),
                                vmin=np.min([np.min(i) for i in self.grid_potentials_history]),
                                vmax=np.max([np.max(i) for i in self.grid_potentials_history]),
                                interpolation="bicubic", origin="lower",
                                extent=[0, self.x_length, 0, self.y_length])

        self.timetext = ax1.text(self.x_length / 8, self.y_length * 0.5, "0")

        self.animation_object = animation.FuncAnimation(fig, self.animate_history,
                                                   frames=len(self.grid_potentials_history),
                                                   save_count=len(self.grid_potentials_history),
                                                   interval=1, blit=True)
        plt.show()


