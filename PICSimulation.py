import numpy as np
import matplotlib.colors as colours
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as time
from functools import partial
from ParticleManager import ParticleManager
from ParticleSource import ParticleSource
from Diagnostics import ParticleSystemHistory
from HelperFunctions import*
from BoundaryClasses import*
from ParticleTypes import*


def sample_maxwell_boltzmann_velocity_distribution(v_thermal, M):
    """Returns a velocity taken from the Maxwell-Boltzmann distribution ,given vthermal (rms) https://www.particleincell.com/2011/mcc/
    :param v_thermal: the thermal velocity of the distribution
    :param M: controls the accuracy of the sample, higher M = more accurate"""
    a = np.sum(np.random.rand(M))
    velocity = np.sqrt(M/12) * (a - M/2) * v_thermal

    return velocity


def new_velocity(v_drift, v_thermal, M):
    """Returns random 2d velocity from given thermal velocity, and drift velocity"""
    x_vel = sample_maxwell_boltzmann_velocity_distribution(v_thermal, M)
    y_vel = sample_maxwell_boltzmann_velocity_distribution(v_thermal, M)

    velocity = np.array([[x_vel], [y_vel]]) + v_drift

    return velocity


def thermal_velocity(charge, temperature, mass):
    return np.sqrt(2*charge*temperature/mass)


def debye_length(eps_0, temperature, neutral_density, charge):
    return np.sqrt(eps_0*temperature/neutral_density/charge)


class PICSimulation:
    def __init__(self,
                 delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length,
                 num_time_steps, printout_interval, snapshot_interval,
                 particle_positions=np.array([[], []]),
                 particle_velocities=np.array([[], []]),
                 particle_charges=np.array([]),
                 particle_masses=np.array([]),
                 particle_types=np.array([]),
                 boundary_conditions=[],
                 particle_sources=[],
                 replace_lost_particles=False,
                 integration_method="LEAPFROG",
                 collision_scheme="NONE"):

        self.printout_interval = printout_interval  # timesteps between printouts
        self.snapshot_interval = snapshot_interval  # timesteps between snapshots
        self.num_time_steps = num_time_steps  # num of time steps to simulate for

        self.particle_system = ParticleManager(delta_t, eps_0, num_x_nodes, num_y_nodes,
                                               x_length, y_length,
                                               particle_positions=particle_positions,
                                               particle_velocities=particle_velocities,
                                               particle_charges=particle_charges,
                                               particle_masses=particle_masses,
                                               particle_types=particle_types,
                                               boundary_conditions=boundary_conditions,
                                               particle_sources=particle_sources,
                                               replace_lost_particles=replace_lost_particles,
                                               integration_method=integration_method,
                                               collision_scheme=collision_scheme)

        self.particle_system_history = ParticleSystemHistory(self.particle_system)

    def begin_simulation(self):
        for i in range(self.num_time_steps):
            self.particle_system.update_step()

            # printout system state
            if self.particle_system.current_t_step % self.printout_interval == 0:
                print(f'{self.particle_system.current_t_step / self.num_time_steps * 100:<6.2f}% completion')
                print("Computational Time Elapsed =", time.time() - self.particle_system.start_time, "s")
                average_time_per_step = (time.time() - self.particle_system.start_time) / self.particle_system.current_t_step
                steps_to_go = self.num_time_steps - self.particle_system.current_t_step
                approx_time_to_go = steps_to_go * average_time_per_step
                print("Approx Time To Finish =", round(approx_time_to_go), "s")
                print("T Step =", self.particle_system.current_t_step, "/", self.num_time_steps)
                print("Simulation Time =", self.particle_system.current_t_step * self.particle_system.delta_t, "s")
                print("Num Particles =", self.particle_system.num_particles)
                print(" ")

            # take a system snapshot for animation
            if self.particle_system.current_t_step % self.snapshot_interval == 0:
                self.particle_system_history.take_snapshot()

    def animate_history(self, i):
        # animate particles
        electron_positions = self.particle_system_history.particle_positions_history[i].T[
            self.particle_system_history.particle_types_history[i] == ParticleTypes.ELECTRON].T
        ion_positions = self.particle_system_history.particle_positions_history[i].T[
            self.particle_system_history.particle_types_history[i] == ParticleTypes.ARGON_ION].T
        self.electrons.set_data(electron_positions)
        self.ions.set_data(ion_positions)

        # animate charge density
        self.densities.set_data(self.particle_system_history.grid_charge_densities_history[i].T)

        # set time text
        self.timetext.set_text(f'{self.particle_system_history.simulation_time_history[i] * 1000:<6.4f}ms')

        # animate potentials
        self.potentials.set_array(self.particle_system_history.grid_potentials_history[i].T)

        return self.potentials, self.densities, self.timetext, self.electrons, self.ions

    def begin_animation(self):
        fig = plt.figure(figsize=(6, 8))

        ax3 = fig.add_subplot(311)
        ax3.set_title("Positions")
        ax3.set_xlim(0, self.particle_system.x_length)
        ax3.set_ylim(0, self.particle_system.y_length)
        self.ions, = ax3.plot([], [], 'bo', markersize=0.5)
        self.electrons, = ax3.plot([], [], 'ro', markersize=0.5)

        ax1 = fig.add_subplot(312)
        ax1.set_title("Densities")
        self.densities = ax1.imshow(np.zeros((self.particle_system.num_x_nodes, self.particle_system.num_y_nodes)),
                               vmin=np.min([np.min(i) for i in self.particle_system_history.grid_charge_densities_history]),
                               vmax=np.max([np.max(i) for i in self.particle_system_history.grid_charge_densities_history]),
                               interpolation="bicubic", origin="lower",
                               extent=[0, self.particle_system.x_length, 0, self.particle_system.y_length])

        ax2 = fig.add_subplot(313)
        ax2.set_title("Potential")
        self.potentials = ax2.imshow(np.zeros((self.particle_system.num_x_nodes, self.particle_system.num_y_nodes)),
                                vmin=np.min([np.min(i) for i in self.particle_system_history.grid_potentials_history]),
                                vmax=np.max([np.max(i) for i in self.particle_system_history.grid_potentials_history]),
                                interpolation="bicubic", origin="lower",
                                extent=[0, self.particle_system.x_length, 0, self.particle_system.y_length])

        self.timetext = ax3.text(self.particle_system.x_length / 2, self.particle_system.y_length * 0.9, "0")

        self.animation_object = animation.FuncAnimation(fig, self.animate_history,
                                                   frames=len(self.particle_system_history.grid_potentials_history),
                                                   save_count=len(self.particle_system_history.grid_potentials_history),
                                                   interval=1, blit=True)
        plt.show()

    def diagnostics(self):
        self.particle_system_history.plot_average_potential()
        self.particle_system_history.plot_average_charge_density()
        self.particle_system_history.plot_energy_history()
        # particle_system_history.plot_particle_number()


class BoxDebyeSheathExample(PICSimulation):
    def __init__(self, delta_t, num_time_steps, num_x_nodes, num_y_nodes, x_length, y_length, sp_w, neutral_density, electron_temp, ion_temp):
        # PARAMETERS
        eps_0 = 8.854e-12  # permittivity of free space
        lD = debye_length(eps_0, electron_temp, neutral_density, Electron().charge)
        electron_vth = thermal_velocity(Electron().charge, electron_temp, Electron().mass)
        ion_vth = thermal_velocity(ArgonIon().charge, ion_temp, ArgonIon().mass)

        num_particles = int(2*neutral_density*x_length*y_length/sp_w)
        printout_interval = 500  # timesteps between printouts
        snapshot_interval = 10  # timesteps between snapshots

        particle_positions = np.random.rand(2, num_particles) * [[x_length], [y_length]]
        particle_velocities = (np.random.rand(2, num_particles) - 0.5)
        particle_charges = np.append(np.zeros(int(num_particles / 2)) + ArgonIon().charge * sp_w,
                                     np.zeros(int(num_particles / 2)) + Electron().charge * sp_w)
        particle_masses = np.append(np.zeros(int(num_particles / 2)) + ArgonIon().mass * sp_w,
                                    np.zeros(int(num_particles / 2)) + Electron().mass * sp_w)
        particle_types = np.append([ArgonIon().type] * int(num_particles / 2),
                                   [Electron().type] * int(num_particles / 2))

        left_bc = LeftBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.FLOATING, 0,
                               BoundaryParticleInteraction.DESTROY, collect_charge=True)
        right_bc = RightBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.FLOATING, 0,
                                 BoundaryParticleInteraction.DESTROY, collect_charge=True)
        upper_bc = UpperBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.DIRICHLET, 0,
                                 BoundaryParticleInteraction.REFLECT, collect_charge=False)
        lower_bc = LowerBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.DIRICHLET, 0,
                                 BoundaryParticleInteraction.REFLECT, collect_charge=False)

        super().__init__(delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length,
                 num_time_steps, printout_interval, snapshot_interval,
                 particle_positions=particle_positions,
                 particle_velocities=particle_velocities,
                 particle_charges=particle_charges,
                 particle_masses=particle_masses,
                 particle_types=particle_types,
                 boundary_conditions=[upper_bc, lower_bc, left_bc, right_bc],
                 replace_lost_particles=True)

delta_t = 1e-9  # step size
num_time_steps = 10000
num_x_nodes = 150
num_y_nodes = 10
x_length = 0.015
y_length = 0.001
sp_w = 10000
neutral_density = 1e12

sim = BoxDebyeSheathExample(delta_t, num_time_steps, num_x_nodes, num_y_nodes, x_length, y_length, sp_w, neutral_density)

sim.begin_simulation()
sim.begin_animation()
sim.diagnostics()
