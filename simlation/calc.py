import numpy as np


def displacement(body, other_body):
    """Return the displacement vector from body to other_body in AU."""
    return other_body.position - body.position


def distance(body, other_body):
    """Return the distance between two bodies in AU."""
    return np.linalg.norm(displacement(body, other_body))


def acceleration(body, other_body, gravitational_constant):
    """Return gravitational acceleration in AU/year^2."""
    r_vec = displacement(body, other_body)
    r = np.linalg.norm(r_vec)

    if r == 0:
        raise ValueError("Two bodies cannot occupy the same position.")

    return gravitational_constant * other_body.mass * r_vec / r**3


def update_velocity(body, dt):
    """Update velocity in AU/year."""
    body.velocity += body.acceleration * dt
    return body.velocity


def update_position(body, dt):
    """Update position in AU."""
    body.position += body.velocity * dt
    return body.position


def total_energy(body1, body2, gravitational_constant):
    """Return kinetic, potential, and total energy of a two-body system."""
    r = distance(body1, body2)

    if r == 0:
        raise ValueError("Two bodies cannot occupy the same position.")

    kinetic = 0.5 * body1.mass * np.dot(body1.velocity, body1.velocity)
    kinetic += 0.5 * body2.mass * np.dot(body2.velocity, body2.velocity)
    potential = (-gravitational_constant * body1.mass * body2.mass / r)

    return kinetic, potential, kinetic + potential
