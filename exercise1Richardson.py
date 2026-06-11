import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simlation import G, M_EARTH, M_SUN, Planet
from simlation import eulerRichardson


ORBIT_RADIUS = 1.0  # AU
SIMULATION_TIME = 5.0  # year
TIME_STEPS = (0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001)


def create_sun_earth_system() -> tuple[Planet, Planet]:
    """Create circular initial conditions in the center-of-mass frame."""
    total_mass = M_SUN + M_EARTH
    relative_speed = np.sqrt(G * total_mass / ORBIT_RADIUS)

    sun = Planet(
        position=[-M_EARTH / total_mass * ORBIT_RADIUS, 0.0],
        velocity=[0.0, -M_EARTH / total_mass * relative_speed],
        acceleration=[0.0, 0.0],
        mass=M_SUN,
    )
    earth = Planet(
        position=[M_SUN / total_mass * ORBIT_RADIUS, 0.0],
        velocity=[0.0, M_SUN / total_mass * relative_speed],
        acceleration=[0.0, 0.0],
        mass=M_EARTH,
    )
    return sun, earth


def simulate(dt: float) -> dict[str, np.ndarray | float]:
    """Run one Euler-Richardson simulation."""
    sun, earth = create_sun_earth_system()
    eulerRichardson.DT = dt

    steps = round(SIMULATION_TIME / dt)
    times = np.arange(steps + 1, dtype=float) * dt
    radii = np.empty(steps + 1, dtype=float)

    sample_stride = max(1, steps // 5000)
    orbit = []

    for index in range(steps + 1):
        relative_position = earth.position - sun.position
        radii[index] = np.linalg.norm(relative_position)

        if index % sample_stride == 0 or index == steps:
            orbit.append(relative_position.copy())

        if index < steps:
            eulerRichardson.step(sun, earth)

    return {
        "time": times,
        "radius": radii,
        "orbit": np.asarray(orbit),
        "maximum_radius": float(np.max(radii)),
    }


def save_summary(output_directory: Path, results: dict[float, dict]) -> None:
    """Save maximum-radius results as CSV."""
    output_path = output_directory / "maximum_radius_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_step_year",
                "maximum_radius_AU",
                "increase_from_initial_AU",
                "relative_increase_percent",
            ]
        )

        for dt, result in results.items():
            maximum_radius = result["maximum_radius"]
            increase = maximum_radius - ORBIT_RADIUS
            writer.writerow(
                [
                    dt,
                    maximum_radius,
                    increase,
                    increase / ORBIT_RADIUS * 100.0,
                ]
            )


def plot_maximum_radius(
    output_directory: Path,
    results: dict[float, dict],
) -> None:
    """Plot maximum orbital radius against the time step."""
    time_steps = np.asarray(list(results.keys()))
    maximum_radii = np.asarray(
        [result["maximum_radius"] for result in results.values()]
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.semilogx(
        time_steps,
        maximum_radii,
        marker="o",
        label="Euler-Richardson",
    )
    axis.axhline(
        ORBIT_RADIUS,
        color="black",
        linestyle="--",
        label="Initial radius = 1 AU",
    )
    axis.set_xlabel("Time step dt [year]")
    axis.set_ylabel("Maximum orbital radius [AU]")
    axis.set_title("Maximum orbital radius versus time step")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory / "maximum_radius_vs_timestep.png",
        dpi=200,
    )
    plt.close(figure)


def plot_radius_history(
    output_directory: Path,
    results: dict[float, dict],
) -> None:
    """Plot orbital-radius histories."""
    figure, axis = plt.subplots(figsize=(8, 5))

    for dt, result in results.items():
        axis.plot(
            result["time"],
            result["radius"],
            linewidth=1.0,
            label=f"dt = {dt:g}",
        )

    axis.set_xlabel("Time [year]")
    axis.set_ylabel("Orbital radius [AU]")
    axis.set_title(
        "Orbital-radius change by the Euler-Richardson method"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_directory / "radius_history.png", dpi=200)
    plt.close(figure)


def plot_orbits(
    output_directory: Path,
    results: dict[float, dict],
) -> None:
    """Compare the largest and smallest time-step orbits."""
    figure, axis = plt.subplots(figsize=(7, 7))

    for dt in (TIME_STEPS[0], TIME_STEPS[-1]):
        orbit = results[dt]["orbit"]
        axis.plot(
            orbit[:, 0],
            orbit[:, 1],
            linewidth=1.0,
            label=f"dt = {dt:g}",
        )

    axis.scatter(0.0, 0.0, color="orange", s=70, label="Sun")
    axis.set_xlabel("Relative x [AU]")
    axis.set_ylabel("Relative y [AU]")
    axis.set_title("Euler-Richardson orbit comparison")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "orbit_comparison.png", dpi=200)
    plt.close(figure)


def main() -> None:
    output_directory = (
        Path(__file__).resolve().parent
        / "exercise1_richardson_results"
    )
    output_directory.mkdir(exist_ok=True)

    results = {}

    print("Euler-Richardson maximum orbital radius")
    print(f"simulation time: {SIMULATION_TIME:g} years")
    print()
    print(
        f"{'dt [year]':>12} "
        f"{'maximum radius [AU]':>22} "
        f"{'increase [%]':>16}"
    )

    for dt in TIME_STEPS:
        result = simulate(dt)
        results[dt] = result
        relative_increase = (
            result["maximum_radius"] - ORBIT_RADIUS
        ) / ORBIT_RADIUS * 100.0
        print(
            f"{dt:12.6g} "
            f"{result['maximum_radius']:22.12f} "
            f"{relative_increase:16.8f}"
        )

    save_summary(output_directory, results)
    plot_maximum_radius(output_directory, results)
    plot_radius_history(output_directory, results)
    plot_orbits(output_directory, results)

    print()
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()
