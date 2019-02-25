import numpy as np
import time as time
from functools import partial
from ParticleGridSystem import ParticleGridSystem
from Diagnostics import SimulationHistorian, SimulationDataAnalyser
from ParticleSources import*
from BoundaryClasses import*
from PlasmaPhysicsFunctions import*
import pprint


class PICSimulation:
    """Runs the simulation while periodically recording data and outputting results."""
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

        self.particle_system = ParticleGridSystem(delta_t, eps_0, num_x_nodes, num_y_nodes,
                                                  x_length, y_length,
                                                  particle_positions=particle_positions,
                                                  particle_velocities=particle_velocities,
                                                  particle_charges=particle_charges,
                                                  particle_masses=particle_masses,
                                                  particle_types=particle_types,
                                                  boundary_conditions=boundary_conditions,
                                                  particle_sources=particle_sources,
                                                  integration_method=integration_method,
                                                  collision_scheme=collision_scheme)

        self.simulation_parameters = {
            "Name": self.__class__.__name__,
            "x_length": x_length,
            "y_length": y_length,
            "num_x_nodes": num_x_nodes,
            "num_y_nodes": num_y_nodes,
            "delta_t": delta_t,
            "eps_0": eps_0,
            "delta_x": self.particle_system.delta_x,
            "delta_y": self.particle_system.delta_y,
            "integration_method": integration_method,
            "collision_scheme": collision_scheme,
            "boundary_conditions": [boundary_condition.__class__.__name__ for boundary_condition in boundary_conditions],
            "particle_sources": [particle_source.__class__.__name__ for particle_source in particle_sources],
            "initial_num_particles": [self.particle_system.num_particles]
        }

    def initiate_simulation(self, num_time_steps, printout_interval, snapshot_interval, save_foldername):
        """
        :param num_time_steps: number of time steps to run sim for
        :param printout_interval: number of timesteps between console printouts
        :param snapshot_interval: number of timesteps between recording data
        :param save_foldername: data file name to save to
        """
        simulation_historian = SimulationHistorian(self.particle_system)

        self.simulation_parameters['num_time_steps'] = num_time_steps
        print("SIMULATION CHARACTERISTICS:")
        print(" ")
        pprint.pprint(self.simulation_parameters)
        print(" ")

        for i in range(num_time_steps):
            self.particle_system.update_step()

            # printout system state
            if self.particle_system.current_t_step % printout_interval == 0:
                print(f'{self.particle_system.current_t_step / num_time_steps * 100:<6.2f}% completion')
                print("Computational Time Elapsed =", time.time() - self.particle_system.start_time, "s")
                average_time_per_step = (time.time() - self.particle_system.start_time) / self.particle_system.current_t_step
                steps_to_go = num_time_steps - self.particle_system.current_t_step
                approx_time_to_go = steps_to_go * average_time_per_step
                print("Approx Time To Finish =", round(approx_time_to_go), "s")
                print("T Step =", self.particle_system.current_t_step, "/", num_time_steps)
                print("Simulation Time =", self.particle_system.current_t_step * self.particle_system.delta_t, "s")
                print("Num Particles =", self.particle_system.num_particles)
                print(" ")

            # take a system snapshot for animation
            if self.particle_system.current_t_step % snapshot_interval == 0:
                simulation_historian.take_snapshot()

        del self.particle_system

        simulation_historian.save_results(save_foldername, self.simulation_parameters)


class BoxDebyeSheathExample(PICSimulation):
    """Basic example of debye sheathing in a box. Left and right sides are floating walls that collect charge."""
    def __init__(self, x_length, y_length, specific_weight, neutral_density, electron_temp, ion_temp, num_x_nodes=None, num_y_nodes=None):

        # plasma characteristics
        eps_0 = 8.854e-12  # permittivity of free space
        electron_debye_length = debye_length(eps_0, electron_temp, neutral_density, Electron().charge)
        electron_vth = thermal_velocity(Electron().charge, electron_temp, Electron().mass)
        ion_vth = thermal_velocity(ArgonIon().charge, ion_temp, ArgonIon().mass)

        # number of electrons, and number of ions to create
        num_particles = int(neutral_density * x_length * y_length / specific_weight)

        # must resolve at least Debye length
        min_num_x_nodes = int(x_length / electron_debye_length) + 2
        min_num_y_nodes = int(y_length / electron_debye_length) + 2

        if num_x_nodes is None:
            num_x_nodes = min_num_x_nodes
        elif num_x_nodes < min_num_x_nodes:
            raise ValueError("WARNING: not enough x-nodes to resolve Debye length. Minimum: " + str(min_num_x_nodes))

        if num_y_nodes is None:
            num_y_nodes = min_num_y_nodes
        elif num_y_nodes < min_num_y_nodes:
            raise ValueError("WARNING: not enough y-nodes to resolve Debye length. Minimum: " + str(min_num_y_nodes))

        delta_x = x_length / (num_x_nodes - 1)
        delta_y = y_length / (num_y_nodes - 1)
        delta_t = np.min([delta_x, delta_y]) / electron_vth / 3  # ensure particles only cross 33% of cell width per dt

        particle_positions = np.random.rand(2, num_particles*2) * [[x_length], [y_length]]

        ion_velocities = sample_maxwell_boltzmann_velocity_distribution(ion_vth, num_particles)
        electron_velocities = sample_maxwell_boltzmann_velocity_distribution(electron_vth, num_particles)
        particle_velocities = np.concatenate((ion_velocities, electron_velocities), axis=1)

        particle_charges = np.append(np.zeros(num_particles) + ArgonIon().charge * specific_weight,
                                     np.zeros(num_particles) + Electron().charge * specific_weight)
        particle_masses = np.append(np.zeros(num_particles) + ArgonIon().mass * specific_weight,
                                    np.zeros(num_particles) + Electron().mass * specific_weight)
        particle_types = np.append([ArgonIon().type] * num_particles,
                                   [Electron().type] * num_particles)

        left_bc = LeftBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.FLOATING, 0,
                               BoundaryParticleInteraction.DESTROY, collect_charge=True)
        right_bc = RightBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.FLOATING, 0,
                                 BoundaryParticleInteraction.DESTROY, collect_charge=True)
        upper_bc = UpperBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.DIRICHLET, 0,
                                 BoundaryParticleInteraction.REFLECT, collect_charge=False)
        lower_bc = LowerBoundary(num_x_nodes, num_y_nodes, FieldBoundaryCondition.DIRICHLET, 0,
                                 BoundaryParticleInteraction.REFLECT, collect_charge=False)

        super().__init__(delta_t, eps_0, num_x_nodes, num_y_nodes, x_length, y_length,
                         particle_positions=particle_positions,
                         particle_velocities=particle_velocities,
                         particle_charges=particle_charges,
                         particle_masses=particle_masses,
                         particle_types=particle_types,
                         boundary_conditions=[upper_bc, lower_bc, left_bc, right_bc],
                         particle_sources=[MaintainChargeDensityInArea(num_particles, electron_vth, ion_vth, specific_weight, [0.005, 0], [0.025, 0.01])]
                         )

        # save sim parameters for reproduction
        self.simulation_parameters['specific_weight'] = specific_weight
        self.simulation_parameters['neutral_density'] = neutral_density
        self.simulation_parameters['electron_temp'] = electron_temp
        self.simulation_parameters['ion_temp'] = ion_temp
        self.simulation_parameters['debye_length'] = electron_debye_length
        self.simulation_parameters['electron_vth'] = electron_vth
        self.simulation_parameters['ion_vth'] = ion_vth


x_length = 0.03
y_length = 0.01
num_x_nodes = 30
num_y_nodes = 10

sp_w = 20000
neutral_density = 1e12
electron_temp = 0.5
ion_temp = 0.1

num_time_steps = 20000
printout_interval = num_time_steps/20
snapshot_interval = 20

save_foldername = 'debye_test_4'

sim = BoxDebyeSheathExample(x_length, y_length, sp_w, neutral_density, electron_temp, ion_temp, num_x_nodes, num_y_nodes)
sim.initiate_simulation(num_time_steps, printout_interval, snapshot_interval, save_foldername)
del sim

simulation_data_analyser = SimulationDataAnalyser(save_foldername)
simulation_data_analyser.plot_charge_density_at_point(0, 2)
# simulation_data_analyser.plot_charge_density(1500)
# simulation_data_analyser.plot_simulation_state(5)
simulation_data_analyser.plot_average_particle_speeds([ParticleTypes.ARGON_ION])
simulation_data_analyser.begin_animation()
simulation_data_analyser.plot_average_particle_type_density([ParticleTypes.ELECTRON, ParticleTypes.ARGON_ION])
simulation_data_analyser.plot_average_charge_density()
simulation_data_analyser.plot_average_potential()
simulation_data_analyser.plot_energy_history()
simulation_data_analyser.plot_particle_number([ParticleTypes.ELECTRON, ParticleTypes.ARGON_ION])