import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from simlation import DT, G
from simlation import galaxy


BLACK_HOLE_MASS = 1.0
STAR_MASS = 1.0e-6
INNER_RADIUS = 0.5  # AU
OUTER_RADIUS = 5.0  # AU
RING_COUNT = 10
SOFTENING = 1.0e-3  # AU
BLOCK_SIZE = 128

QUICK_STAR_COUNTS = (1_000, 2_000, 4_000, 8_000)
FULL_STAR_COUNTS = (1_000, 3_000, 10_000, 30_000, 100_000)

PREVIEW_STAR_COUNT = 300
PREVIEW_STEPS = 500


def create_ring_galaxy(
    star_count: int,
    ring_count: int = RING_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a central black hole and stars on circular rings."""
    if star_count < ring_count:
        raise ValueError("Star count must be at least the ring count.")

    radii = np.linspace(INNER_RADIUS, OUTER_RADIUS, ring_count)
    counts = np.full(ring_count, star_count // ring_count, dtype=int)
    counts[: star_count % ring_count] += 1

    star_positions = []
    star_velocities = []
    ring_ids = []
    enclosed_star_count = 0

    for ring_index, (radius, count) in enumerate(zip(radii, counts)):
        phase = ring_index * np.pi / ring_count
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        angles += phase

        positions = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles))
        )

        enclosed_mass = BLACK_HOLE_MASS + enclosed_star_count * STAR_MASS
        circular_speed = np.sqrt(G * enclosed_mass / radius)
        velocities = np.column_stack(
            (
                -circular_speed * np.sin(angles),
                circular_speed * np.cos(angles),
            )
        )

        star_positions.append(positions)
        star_velocities.append(velocities)
        ring_ids.append(np.full(count, ring_index, dtype=int))
        enclosed_star_count += count

    star_positions = np.vstack(star_positions)
    star_velocities = np.vstack(star_velocities)
    ring_ids = np.concatenate(ring_ids)

    positions = np.vstack(([0.0, 0.0], star_positions))
    velocities = np.vstack(([0.0, 0.0], star_velocities))
    masses = np.concatenate(
        ([BLACK_HOLE_MASS], np.full(star_count, STAR_MASS))
    )

    # Remove the small residual center-of-mass velocity.
    velocities[0] = -np.sum(
        masses[1:, np.newaxis] * velocities[1:],
        axis=0,
    ) / BLACK_HOLE_MASS

    return positions, velocities, masses, ring_ids


def benchmark_one_step(star_counts: tuple[int, ...]) -> list[tuple[int, float]]:
    """Measure the elapsed time of one exact N-body step."""
    timings = []

    for star_count in star_counts:
        positions, velocities, masses, _ = create_ring_galaxy(star_count)

        start = perf_counter()
        galaxy.step(
            positions,
            velocities,
            masses,
            dt=DT,
            softening=SOFTENING,
            block_size=BLOCK_SIZE,
        )
        elapsed = perf_counter() - start
        timings.append((star_count, elapsed))
        print(f"{star_count:>10,d} stars : {elapsed:12.6f} seconds")

    return timings


def simulate_preview():
    """Run a small galaxy simulation for a visual check."""
    positions, velocities, masses, ring_ids = create_ring_galaxy(
        PREVIEW_STAR_COUNT
    )
    initial_positions = positions.copy()

    for _ in range(PREVIEW_STEPS):
        galaxy.step(
            positions,
            velocities,
            masses,
            dt=DT,
            softening=SOFTENING,
            block_size=BLOCK_SIZE,
        )

    return initial_positions, positions, ring_ids


def save_timings(
    output_directory: Path,
    timings: list[tuple[int, float]],
) -> None:
    """Save timing data as CSV."""
    output_path = output_directory / "galaxy_timing_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["star_count", "one_step_seconds"])
        writer.writerows(timings)


def plot_timings(
    output_directory: Path,
    timings: list[tuple[int, float]],
) -> None:
    """Plot measured time and an N-squared reference line."""
    star_counts = np.asarray([item[0] for item in timings], dtype=float)
    elapsed_times = np.asarray([item[1] for item in timings], dtype=float)
    reference = elapsed_times[0] * (star_counts / star_counts[0]) ** 2

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.loglog(
        star_counts,
        elapsed_times,
        marker="o",
        label="Measured",
    )
    axis.loglog(
        star_counts,
        reference,
        linestyle="--",
        label=r"$N^2$ reference",
    )
    axis.set_xlabel("Number of stars")
    axis.set_ylabel("Time per step [s]")
    axis.set_title("Direct N-body calculation time")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory / "calculation_time_vs_star_count.png",
        dpi=200,
    )
    plt.close(figure)


def plot_preview(
    output_directory: Path,
    initial_positions: np.ndarray,
    final_positions: np.ndarray,
    ring_ids: np.ndarray,
) -> None:
    """Plot the initial and final ring distribution."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))

    for axis, positions, title in (
        (axes[0], initial_positions, "Initial state"),
        (
            axes[1],
            final_positions,
            f"After {PREVIEW_STEPS} steps",
        ),
    ):
        axis.scatter(
            positions[1:, 0],
            positions[1:, 1],
            c=ring_ids,
            cmap="viridis",
            s=8,
        )
        axis.scatter(
            positions[0, 0],
            positions[0, 1],
            color="black",
            s=80,
            label="Black hole",
        )
        axis.set_title(title)
        axis.set_xlabel("x [AU]")
        axis.set_ylabel("y [AU]")
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("Galaxy simulation with direct star-star gravity")
    figure.tight_layout()
    figure.savefig(output_directory / "galaxy_preview.png", dpi=200)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark a direct N-body galaxy simulation."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include up to 100,000 stars. This may take a very long time.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the small visual simulation.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    star_counts = FULL_STAR_COUNTS if arguments.full else QUICK_STAR_COUNTS

    output_directory = (
        Path(__file__).resolve().parent / "exercise2_galaxy_results"
    )
    output_directory.mkdir(exist_ok=True)

    print("Direct N-body galaxy simulation")
    print(f"block size: {BLOCK_SIZE}")
    print()

    timings = benchmark_one_step(star_counts)
    save_timings(output_directory, timings)
    plot_timings(output_directory, timings)

    if not arguments.no_preview:
        initial_positions, final_positions, ring_ids = simulate_preview()
        plot_preview(
            output_directory,
            initial_positions,
            final_positions,
            ring_ids,
        )

    print()
    print(f"Results saved in: {output_directory}")
    if not arguments.full:
        print("Use --full to benchmark up to 100,000 stars.")


if __name__ == "__main__":
    main()
