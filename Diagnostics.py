import numpy as np
import matplotlib.pyplot as plt
from PlasmaPhysicsFunctions import ParticleTypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os
import pickle
import itertools

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

    def save_results(self, foldername, simulation_parameters=None):
        """Save simulation data"""
        del self.particle_system

        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(script_dir, foldername)

        try:
            os.makedirs(folder)
        except OSError:
            pass  # already exists

        #TODO: doesnt work with changing number of particles
        with open(os.path.join(folder, 'data.pkl'), 'wb') as file:
            pickle.dump(self, file)
        with open(os.path.join(folder, 'parameters.pkl'), 'wb') as file:
            pickle.dump(simulation_parameters, file)
        with open(os.path.join(folder, 'parameters.txt'), 'w') as file:
            file.write(json.dumps(simulation_parameters))

        print("Data successfully saved to folder", foldername)


class SimulationDataAnalyser:
    def __init__(self, foldername):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder = os.path.join(script_dir, foldername)

        with open(os.path.join(self.folder, 'data.pkl'), 'rb') as file:
            data = pickle.load(file)
            self.simulation_time_history = data.simulation_time_history

            self.grid_potentials_history = data.grid_potentials_history
            self.grid_charge_densities_history = data.grid_charge_densities_history

            self.particle_positions_history = data.particle_positions_history
            self.particle_velocities_history = data.particle_velocities_history
            self.particle_types_history = data.particle_types_history
            self.particle_masses_history = data.particle_masses_history

        with open(os.path.join(self.folder, 'parameters.pkl'), 'rb') as file:
            parameters = pickle.load(file)
            self.num_x_nodes = parameters['num_x_nodes']
            self.num_y_nodes = parameters['num_y_nodes']
            self.delta_x = parameters['delta_x']
            self.delta_y = parameters['delta_y']
            self.x_length = parameters['x_length']
            self.y_length = parameters['y_length']

        self.num_time_steps = len(self.simulation_time_history)

    def plot_energy_history(self):
        """Calculates and plots stored energy histories"""
        energy_potential_history = []
        energy_kinetic_history = []
        energy_total_history = []

        for i in range(self.num_time_steps):
            total_potential = np.sum(1 / 2 * self.grid_potentials_history[i] * self.grid_charge_densities_history[i] *
                                     self.delta_x * self.delta_y)

            total_kinetic = np.sum(1 / 2 * self.particle_masses_history[i] *
                                   np.linalg.norm(self.particle_velocities_history[i].T, axis=1) ** 2)

            total_energy = total_kinetic + total_potential

            energy_potential_history = energy_potential_history + [total_potential]
            energy_kinetic_history = energy_kinetic_history + [total_kinetic]
            energy_total_history = energy_total_history + [total_energy]

        plt.plot(self.simulation_time_history, energy_potential_history, label='potential')
        plt.plot(self.simulation_time_history, energy_kinetic_history, label='kinetic')
        plt.plot(self.simulation_time_history, energy_total_history, label='total')
        plt.legend(loc='lower right')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Energy over time')
        plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        plt.show()

    def average_list_of_arrays(self, list_of_arrays):
        avr = np.zeros(np.shape(list_of_arrays[0]))
        for array in list_of_arrays:
            avr += array
        avr /= len(list_of_arrays)

        return avr

    def plot_average_potential(self):
        avr_grid_potential = self.average_list_of_arrays(self.grid_potentials_history)

        # plt.imshow(avr_grid_potential.T,
        #            vmin=np.min(avr_grid_potential),
        #            vmax=np.max(avr_grid_potential),
        #            interpolation="bicubic", origin="lower",
        #            extent=[0, self.x_length, 0, self.y_length])
        # plt.title("Averaged System Potential")
        # plt.show()

        positions = np.arange(0, self.num_x_nodes) * self.delta_x

        with open(os.path.join(self.folder, 'potentials.csv'), 'wb') as file:
            np.savetxt(file, [positions, avr_grid_potential[:, int(self.num_y_nodes/2)]], delimiter=",")

        plt.plot(positions, avr_grid_potential[:, int(self.num_y_nodes/2)])
        plt.title("Averaged potential along x")
        plt.show()

    def plot_potential(self, timestep):
        positions = np.arange(0, self.num_x_nodes) * self.delta_x

        with open(os.path.join(self.folder, 'potentials.csv'), 'wb') as file:
            np.savetxt(file, [positions, self.grid_potentials_history[timestep][:, int(self.num_y_nodes/2)]], delimiter=",")

        plt.plot(positions, self.grid_potentials_history[timestep][:, int(self.num_y_nodes/2)])
        plt.title("Averaged potential along x")
        plt.show()

    def plot_charge_density_at_point(self, node_x, node_y):
        charge_density_history = [charge_densities[node_x, node_y] for charge_densities in self.grid_charge_densities_history]
        plt.plot(self.simulation_time_history, charge_density_history)
        plt.show()

    def plot_average_charge_density(self):
        avr_grid_charge_density = self.average_list_of_arrays(self.grid_charge_densities_history)

        positions = np.arange(0, self.num_x_nodes) * self.delta_x

        with open(os.path.join(self.folder, 'densities.csv'), 'wb') as file:
            np.savetxt(file, [positions, avr_grid_charge_density[:, int(self.num_y_nodes/2)]], delimiter=",")

        plt.plot(positions, avr_grid_charge_density[:, int(self.num_y_nodes/2)])
        plt.title("Averaged charge density along x")
        plt.show()

    def plot_charge_density(self, timestep):
        positions = np.arange(0, self.num_x_nodes) * self.delta_x

        with open(os.path.join(self.folder, 'potentials.csv'), 'wb') as file:
            np.savetxt(file, [positions, self.grid_charge_densities_history[timestep][:, int(self.num_y_nodes/2)]], delimiter=",")

        plt.plot(positions, self.grid_charge_densities_history[timestep][:, int(self.num_y_nodes/2)])
        plt.title("Averaged potential along x")
        plt.show()

    def plot_average_particle_type_density(self, particle_types_to_plot):
        for particle_type_to_plot in particle_types_to_plot:
            avr_density = np.zeros(self.num_x_nodes)
            for i, particle_positions in enumerate(self.particle_positions_history):
                # animate density along x
                x_positions = particle_positions[:, self.particle_types_history[i] == particle_type_to_plot][0].flatten()
                cell_length_position = x_positions / self.delta_x
                node_positions = np.floor(cell_length_position).astype(int)
                fractional_positions = (cell_length_position - node_positions)
                particle_densities = np.zeros(self.num_x_nodes)

                np.add.at(particle_densities, node_positions+1, fractional_positions)
                np.add.at(particle_densities, node_positions, 1-fractional_positions)
                density = particle_densities / self.delta_x

                avr_density += density

            avr_density /= self.num_time_steps
            positions = np.arange(0, self.num_x_nodes) * self.delta_x

            with open(os.path.join(self.folder, particle_type_to_plot.name + '.csv'), 'wb') as file:
                np.savetxt(file, [positions, avr_density], delimiter=",")

            plt.plot(positions, avr_density, 'o', label=particle_type_to_plot.name)

        plt.title("Averaged particle density along x")
        plt.legend()
        plt.show()

    def plot_average_particle_speeds(self, particle_types_to_plot):
        for particle_type_to_plot in particle_types_to_plot:
            avr_speeds = np.zeros(self.num_x_nodes)
            for i, particle_positions in enumerate(self.particle_positions_history):
                # animate density along x
                particle_speeds = self.particle_velocities_history[i][:, self.particle_types_history[i] == particle_type_to_plot][0].flatten()
                x_positions = particle_positions[:, self.particle_types_history[i] == particle_type_to_plot][0].flatten()
                cell_length_position = x_positions / self.delta_x
                node_positions = np.round(cell_length_position).astype(int)
                speeds = np.zeros(self.num_x_nodes)

                bin = np.bincount(node_positions, minlength=self.num_x_nodes)

                np.add.at(speeds, node_positions, particle_speeds)
                speeds /= bin
                avr_speeds += speeds

            avr_speeds /= self.num_time_steps
            positions = np.arange(0, self.num_x_nodes) * self.delta_x

            with open(os.path.join(self.folder, 'particle_speeds.csv'), 'wb') as file:
                np.savetxt(file, [positions, avr_speeds], delimiter=",")

            plt.plot(positions, avr_speeds, 'o', label=particle_type_to_plot.name)

        plt.title("Averaged particle x speed along x")
        plt.legend()
        plt.show()

    def plot_particle_number(self, particle_types_to_plot):
        for particle_type_to_plot in particle_types_to_plot:
            count_history = [np.sum(particle_types == particle_type_to_plot) for particle_types in self.particle_types_history]
            plt.plot(count_history, label=particle_type_to_plot.name)

        plt.title("Particle type count over time")
        plt.legend()
        plt.show()

    def plot_simulation_state(self, timestep):
        fig = plt.figure(figsize=(6, 8))

        ax3 = fig.add_subplot(311)
        ax3.set_title("Positions")
        ax3.set_xlim(0, self.x_length)
        ax3.set_ylim(0, self.y_length)

        electron_positions = self.particle_positions_history[timestep].T[
            self.particle_types_history[timestep] == ParticleTypes.ELECTRON].T
        ion_positions = self.particle_positions_history[timestep].T[
            self.particle_types_history[timestep] == ParticleTypes.ARGON_ION].T

        ions = ax3.plot(ion_positions[0], ion_positions[1], 'co', markersize=0.5, alpha=0.5)
        electrons = ax3.plot(electron_positions[0], electron_positions[1], 'ro', markersize=0.5, alpha=0.5)

        ax1 = fig.add_subplot(312)
        ax1.set_title("Densities")
        densities = ax1.imshow(np.zeros((self.num_x_nodes, self.num_y_nodes)),
                                    vmin=np.min([np.min(i) for i in self.grid_charge_densities_history]),
                                    vmax=np.max([np.max(i) for i in self.grid_charge_densities_history]),
                                    interpolation="bicubic", origin="lower",
                                    extent=[0, self.x_length, 0, self.y_length])
        densities.set_data(self.grid_charge_densities_history[timestep].T)

        ax2 = fig.add_subplot(313)
        ax2.set_title("Potential")
        potentials = ax2.imshow(np.zeros((self.num_x_nodes, self.num_y_nodes)),
                                     vmin=np.min([np.min(i) for i in self.grid_potentials_history]),
                                     vmax=np.max([np.max(i) for i in self.grid_potentials_history]),
                                     interpolation="bicubic", origin="lower",
                                     extent=[0, self.x_length, 0, self.y_length])
        potentials.set_array(self.grid_potentials_history[timestep].T)

        timetext = ax1.text(self.x_length / 8, self.y_length * 0.5, "0")
        timetext.set_text(f'{self.simulation_time_history[timestep] * 1000000:<6.4f}µs')

        plt.show()

    def animate_history(self, i):
        # animate particles
        electron_positions = self.particle_positions_history[i].T[self.particle_types_history[i] == ParticleTypes.ELECTRON].T
        ion_positions = self.particle_positions_history[i].T[self.particle_types_history[i] == ParticleTypes.ARGON_ION].T
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
                                                        frames=self.num_time_steps,
                                                        save_count=self.num_time_steps,
                                                        interval=1, blit=True)
        plt.show()


