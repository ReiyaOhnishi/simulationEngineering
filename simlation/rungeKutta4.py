from . import DT, G, Planet
from . import calc


def _temporary_body(
    body: Planet,
    position,
    velocity,
) -> Planet:
    """Create a temporary body used to evaluate an RK4 stage."""
    return Planet(
        position=position,
        velocity=velocity,
        acceleration=body.acceleration,
        mass=body.mass,
    )


def _derivatives(body1: Planet, body2: Planet):
    """Return position and velocity derivatives for both bodies."""
    return (
        body1.velocity,
        calc.acceleration(body1, body2, G),
        body2.velocity,
        calc.acceleration(body2, body1, G),
    )


def step(body1: Planet, body2: Planet) -> tuple[Planet, Planet]:
    """Advance a two-body system by one fourth-order Runge-Kutta step."""
    position1 = body1.position.copy()
    velocity1 = body1.velocity.copy()
    position2 = body2.position.copy()
    velocity2 = body2.velocity.copy()

    k1_r1, k1_v1, k1_r2, k1_v2 = _derivatives(body1, body2)

    stage1 = _temporary_body(
        body1,
        position1 + k1_r1 * (DT / 2.0),
        velocity1 + k1_v1 * (DT / 2.0),
    )
    stage2 = _temporary_body(
        body2,
        position2 + k1_r2 * (DT / 2.0),
        velocity2 + k1_v2 * (DT / 2.0),
    )
    k2_r1, k2_v1, k2_r2, k2_v2 = _derivatives(stage1, stage2)

    stage1 = _temporary_body(
        body1,
        position1 + k2_r1 * (DT / 2.0),
        velocity1 + k2_v1 * (DT / 2.0),
    )
    stage2 = _temporary_body(
        body2,
        position2 + k2_r2 * (DT / 2.0),
        velocity2 + k2_v2 * (DT / 2.0),
    )
    k3_r1, k3_v1, k3_r2, k3_v2 = _derivatives(stage1, stage2)

    stage1 = _temporary_body(
        body1,
        position1 + k3_r1 * DT,
        velocity1 + k3_v1 * DT,
    )
    stage2 = _temporary_body(
        body2,
        position2 + k3_r2 * DT,
        velocity2 + k3_v2 * DT,
    )
    k4_r1, k4_v1, k4_r2, k4_v2 = _derivatives(stage1, stage2)

    body1.position = position1 + DT * (
        k1_r1 + 2.0 * k2_r1 + 2.0 * k3_r1 + k4_r1
    ) / 6.0
    body1.velocity = velocity1 + DT * (
        k1_v1 + 2.0 * k2_v1 + 2.0 * k3_v1 + k4_v1
    ) / 6.0
    body2.position = position2 + DT * (
        k1_r2 + 2.0 * k2_r2 + 2.0 * k3_r2 + k4_r2
    ) / 6.0
    body2.velocity = velocity2 + DT * (
        k1_v2 + 2.0 * k2_v2 + 2.0 * k3_v2 + k4_v2
    ) / 6.0

    # Keep acceleration consistent with the updated positions.
    acceleration1 = calc.acceleration(body1, body2, G)
    acceleration2 = calc.acceleration(body2, body1, G)
    body1.acceleration = acceleration1
    body2.acceleration = acceleration2

    return body1, body2
