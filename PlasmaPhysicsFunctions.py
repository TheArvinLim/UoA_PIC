from enum import Enum
import scipy.stats as stats
import numpy as np
# This file stores all relevant values and functions for plasma physics


class ParticleTypes(Enum):
    """Stores all possible particle types"""
    ELECTRON = 1
    ARGON_ION = 2


class Particle:
    def __init__(self, type, mass, charge):
        self.type = type
        self.mass = mass
        self.charge = charge


class Electron(Particle):
    def __init__(self):
        super().__init__(ParticleTypes.ELECTRON, 9.109e-31, -1.602e-19)


class ArgonIon(Particle):
    def __init__(self):
        super().__init__(ParticleTypes.ARGON_ION, 32*1.661e-27, 1.602e-19)


def sample_maxwell_boltzmann_velocity_distribution(v_thermal, num_velocities):
    """
    Returns an array of velocities sampled from the maxwell boltzmann distribution
    :param v_thermal: thermal velocity (most probable) of the distribution
    :param num_velocities: number of velocities to generate
    :returns: 2 by num_velocities array of velocities
    """
    a = v_thermal / np.sqrt(2)  # shape parameter of distribution

    maxwell = stats.maxwell

    speeds = maxwell.rvs(loc=0, scale=a, size=num_velocities)  # generate speeds
    theta = np.random.rand(num_velocities) * 2 * np.pi  # select random angle

    x_vels = speeds * np.sin(theta)
    y_vels = speeds * np.cos(theta)

    return np.stack((x_vels, y_vels))


def thermal_velocity(charge, temperature, mass):
    """
    Most probable speed in the velocity distribution = sqrt(2eT/m)
    :param charge: particle charge in C
    :param temperature: particle temp in eV
    :param mass: particle mass in kg
    :return: thermal velocity
    """
    return np.sqrt(2*abs(charge)*temperature/mass)


def debye_length(eps_0, electron_temperature, electron_density, elementary_charge):
    """
    Calculate plasma debye length
    :param eps_0: permittivity of free space
    :param electron_temperature: electron temp in eV
    :param electron_density: electron density in particles/volume
    :param elementary_charge: elementary charge in C
    :return: debye length
    """
    return np.sqrt(eps_0*electron_temperature/electron_density/abs(elementary_charge))


def sinusoidal(amplitude, period, phase, t):
    return np.sin(2*np.pi*(t/period) + phase) * amplitude