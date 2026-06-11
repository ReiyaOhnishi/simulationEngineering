from . import DT, G, Planet
from . import calc


def step(body1: Planet, body2: Planet) -> tuple[Planet, Planet]:
    """Advance a two-body system by one Euler-Cromer step."""
    body1.acceleration = calc.acceleration(body1, body2, G)
    body2.acceleration = calc.acceleration(body2, body1, G)

    calc.update_velocity(body1, DT)
    calc.update_velocity(body2, DT)
    calc.update_position(body1, DT)
    calc.update_position(body2, DT)

    return body1, body2
