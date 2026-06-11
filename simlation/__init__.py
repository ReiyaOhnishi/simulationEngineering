from dataclasses import dataclass

import numpy as np


# Unit system:
#   distance = astronomical unit (AU)
#   time     = year
#   mass     = solar mass
G = 4.0 * np.pi**2
DT = 0.001

M_SUN = 1.0
M_EARTH = 3.003e-6


@dataclass
class Planet:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    mass: float

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.acceleration = np.asarray(self.acceleration, dtype=float)
        self.mass = float(self.mass)

        if self.position.shape != self.velocity.shape:
            raise ValueError("Position and velocity must have the same shape.")
        if self.position.shape != self.acceleration.shape:
            raise ValueError("Position and acceleration must have the same shape.")
        if self.mass <= 0:
            raise ValueError("Mass must be positive.")
