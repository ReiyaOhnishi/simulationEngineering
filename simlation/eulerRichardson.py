from . import DT, G, Planet
from . import calc


def step(body1: Planet, body2: Planet) -> tuple[Planet, Planet]:
    """Advance a two-body system by one Euler-Richardson step."""
    acceleration1 = calc.acceleration(body1, body2, G)
    acceleration2 = calc.acceleration(body2, body1, G)

    midpoint1 = Planet(
        position=body1.position + body1.velocity * (DT / 2.0),
        velocity=body1.velocity + acceleration1 * (DT / 2.0),
        acceleration=acceleration1,
        mass=body1.mass,
    )
    midpoint2 = Planet(
        position=body2.position + body2.velocity * (DT / 2.0),
        velocity=body2.velocity + acceleration2 * (DT / 2.0),
        acceleration=acceleration2,
        mass=body2.mass,
    )

    midpoint_acceleration1 = calc.acceleration(midpoint1, midpoint2, G)
    midpoint_acceleration2 = calc.acceleration(midpoint2, midpoint1, G)

    body1.position += midpoint1.velocity * DT
    body2.position += midpoint2.velocity * DT
    body1.velocity += midpoint_acceleration1 * DT
    body2.velocity += midpoint_acceleration2 * DT
    body1.acceleration = midpoint_acceleration1
    body2.acceleration = midpoint_acceleration2

    return body1, body2
