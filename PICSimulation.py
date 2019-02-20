import time as time
from functools import partial
from ParticleGridSystem import ParticleGridSystem
from Diagnostics import SimulationAnalysis
from ParticleSources import*
from BoundaryClasses import*
from PlasmaPhysicsFunctions import*


class PICSimulation:
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

        self.simulation_analysis = SimulationAnalysis(self.particle_system)

    def begin_simulation(self, num_time_steps, printout_interval, snapshot_interval):
        print('SIMULATION START')

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
                self.simulation_analysis.take_snapshot()

    def animate(self):
        self.simulation_analysis.begin_animation()

    def diagnostics(self):
        self.simulation_analysis.plot_average_potential()
        self.simulation_analysis.plot_average_charge_density()
        self.simulation_analysis.plot_energy_history()
        self.simulation_analysis.plot_particle_number()
        self.simulation_analysis.save_results('data')


class BoxDebyeSheathExample(PICSimulation):
    def __init__(self, x_length, y_length, specific_weight, neutral_density, electron_temp, ion_temp, num_x_nodes=None, num_y_nodes=None):
        # PARAMETERS
        eps_0 = 8.854e-12  # permittivity of free space
        electron_debye_length = debye_length(eps_0, electron_temp, neutral_density, Electron().charge)
        electron_vth = thermal_velocity(Electron().charge, electron_temp, Electron().mass)
        ion_vth = thermal_velocity(ArgonIon().charge, ion_temp, ArgonIon().mass)

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
        delta_t = np.min([delta_x, delta_y]) / electron_vth / 10  # ensure particles only cross 10% of cell width per dt

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
                 particle_sources=[MaintainChargeDensity(particle_types, electron_vth, ion_vth, specific_weight)])

        print('SIMULATION CHARACTERISTICS')
        print(" ")
        print("Number of particles =", self.particle_system.num_particles)
        print("dt =", self.particle_system.delta_t, "s")
        print("dx =", self.particle_system.delta_x, "m")
        print("dy =", self.particle_system.delta_y, "m")
        print("num_x_nodes =", self.particle_system.num_x_nodes)
        print("num_y_nodes =", self.particle_system.num_y_nodes)
        print("x_length =", self.particle_system.x_length, "m")
        print("y_length =", self.particle_system.y_length, "m")
        print("Debye length =", electron_debye_length, "m")
        print("Electron thermal velocity =", electron_vth, "m/s")
        print("Ion thermal velocity =", ion_vth, "m/s")
        print(" ")


x_length = 0.15
y_length = 0.01
num_x_nodes = 150
num_y_nodes = 10

sp_w = 100000
neutral_density = 1e12
electron_temp = 1
ion_temp = 0.1

num_time_steps = 2000
printout_interval = num_time_steps/20
snapshot_interval = 10

sim = BoxDebyeSheathExample(x_length, y_length, sp_w, neutral_density, electron_temp, ion_temp, num_x_nodes, num_y_nodes)
sim.begin_simulation(num_time_steps, printout_interval, snapshot_interval)
sim.animate()
sim.diagnostics()
