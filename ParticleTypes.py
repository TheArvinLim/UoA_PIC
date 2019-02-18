from enum import Enum

class ParticleTypes(Enum):
    """Stores all possible particle types"""
    ELECTRON = 'Electron'  # sets value at boundary to constant
    ARGON_ION = 'Argon Ion'  # sets derivative at boundary to constant

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

