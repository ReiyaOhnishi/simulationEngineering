import numpy as np

from . import DT, G


def accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float = G,
    softening: float = 1.0e-3,
    block_size: int = 128,
) -> np.ndarray:
    """Calculate exact direct-sum gravitational accelerations."""
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if positions.ndim != 2:
        raise ValueError("Positions must have shape (body_count, dimensions).")
    if masses.shape != (positions.shape[0],):
        raise ValueError("Masses must have one value for each body.")
    if np.any(masses <= 0):
        raise ValueError("All masses must be positive.")
    if softening < 0:
        raise ValueError("Softening must not be negative.")
    if block_size <= 0:
        raise ValueError("Block size must be positive.")

    body_count = positions.shape[0]
    result = np.empty_like(positions)
    softening_squared = softening**2

    # A block is compared with every body. This keeps memory use O(BN)
    # while retaining the exact O(N^2) direct-force calculation.
    for start in range(0, body_count, block_size):
        end = min(start + block_size, body_count)
        displacement = (
            positions[np.newaxis, :, :]
            - positions[start:end, np.newaxis, :]
        )
        distance_squared = np.einsum(
            "bnd,bnd->bn",
            displacement,
            displacement,
        )
        distance_squared += softening_squared

        local_indices = np.arange(end - start)
        distance_squared[local_indices, start + local_indices] = np.inf

        inverse_distance_cubed = distance_squared**-1.5
        weights = inverse_distance_cubed * masses[np.newaxis, :]
        result[start:end] = gravitational_constant * np.einsum(
            "bnd,bn->bd",
            displacement,
            weights,
            optimize=True,
        )

    return result


def step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float = DT,
    gravitational_constant: float = G,
    softening: float = 1.0e-3,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance all bodies by one Euler-Cromer step."""
    acceleration = accelerations(
        positions,
        masses,
        gravitational_constant=gravitational_constant,
        softening=softening,
        block_size=block_size,
    )
    velocities += acceleration * dt
    positions += velocities * dt
    return positions, velocities, acceleration
