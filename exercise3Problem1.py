import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simlation import G, M_EARTH, M_SUN, Planet
from simlation import calc, eulerRichardson, rungeKutta4


ORBIT_RADIUS = 1.0  # AU
SIMULATION_TIME = 5.0  # year
TIME_STEPS = (0.01, 0.001, 0.0001)
ERROR_FLOOR = 1.0e-18

METHODS = {
    "Euler-Richardson": eulerRichardson,
    "Runge-Kutta 4": rungeKutta4,
}


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


def total_angular_momentum(body1: Planet, body2: Planet) -> float:
    """Return the z component of the system's angular momentum."""
    angular_momentum1 = body1.mass * (
        body1.position[0] * body1.velocity[1]
        - body1.position[1] * body1.velocity[0]
    )
    angular_momentum2 = body2.mass * (
        body2.position[0] * body2.velocity[1]
        - body2.position[1] * body2.velocity[0]
    )
    return float(angular_momentum1 + angular_momentum2)


def simulate(method_module, dt: float) -> dict[str, np.ndarray | float]:
    """Run one method and record conservation-law errors."""
    sun, earth = create_sun_earth_system()
    method_module.DT = dt

    steps = round(SIMULATION_TIME / dt)
    times = np.arange(steps + 1, dtype=float) * dt
    energy_errors = np.empty(steps + 1, dtype=float)
    angular_momentum_errors = np.empty(steps + 1, dtype=float)

    sample_stride = max(1, steps // 5000)
    orbit = []

    initial_energy = calc.total_energy(sun, earth, G)[2]
    initial_angular_momentum = total_angular_momentum(sun, earth)

    for index in range(steps + 1):
        energy = calc.total_energy(sun, earth, G)[2]
        angular_momentum = total_angular_momentum(sun, earth)

        energy_errors[index] = (
            energy - initial_energy
        ) / abs(initial_energy)
        angular_momentum_errors[index] = (
            angular_momentum - initial_angular_momentum
        ) / abs(initial_angular_momentum)

        if index % sample_stride == 0 or index == steps:
            orbit.append((earth.position - sun.position).copy())

        if index < steps:
            method_module.step(sun, earth)

    return {
        "time": times,
        "energy_error": energy_errors,
        "angular_momentum_error": angular_momentum_errors,
        "orbit": np.asarray(orbit),
        "initial_energy": initial_energy,
        "initial_angular_momentum": initial_angular_momentum,
        "maximum_energy_error": float(np.max(np.abs(energy_errors))),
        "maximum_angular_momentum_error": float(
            np.max(np.abs(angular_momentum_errors))
        ),
    }


def save_summary(output_directory: Path, results: dict) -> None:
    """Save maximum relative errors as CSV."""
    output_path = output_directory / "conservation_error_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "method",
                "time_step_year",
                "initial_total_energy",
                "initial_angular_momentum",
                "maximum_relative_energy_error",
                "maximum_relative_angular_momentum_error",
            ]
        )

        for method_name, method_results in results.items():
            for dt, result in method_results.items():
                writer.writerow(
                    [
                        method_name,
                        dt,
                        result["initial_energy"],
                        result["initial_angular_momentum"],
                        result["maximum_energy_error"],
                        result["maximum_angular_momentum_error"],
                    ]
                )


def plot_error_history(
    output_directory: Path,
    results: dict,
    result_key: str,
    output_name: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot one conservation-law error for both methods."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
    )

    for axis, (method_name, method_results) in zip(axes, results.items()):
        for dt, result in method_results.items():
            absolute_error = np.maximum(
                np.abs(result[result_key]),
                ERROR_FLOOR,
            )
            axis.semilogy(
                result["time"],
                absolute_error,
                linewidth=1.0,
                label=f"dt = {dt:g}",
            )

        axis.set_title(method_name)
        axis.set_xlabel("Time [year]")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    axes[0].set_ylabel(ylabel)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_directory / output_name, dpi=200)
    plt.close(figure)


def plot_maximum_errors(output_directory: Path, results: dict) -> None:
    """Plot maximum conservation errors against time step."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    settings = (
        ("maximum_energy_error", "Maximum relative energy error"),
        (
            "maximum_angular_momentum_error",
            "Maximum relative angular momentum error",
        ),
    )

    for axis, (result_key, ylabel) in zip(axes, settings):
        for method_name, method_results in results.items():
            time_steps = np.asarray(list(method_results.keys()))
            errors = np.asarray(
                [
                    max(result[result_key], ERROR_FLOOR)
                    for result in method_results.values()
                ]
            )
            axis.loglog(
                time_steps,
                errors,
                marker="o",
                label=method_name,
            )

        axis.set_xlabel("Time step dt [year]")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.suptitle("Conservation error versus time step")
    figure.tight_layout()
    figure.savefig(
        output_directory / "maximum_error_vs_timestep.png",
        dpi=200,
    )
    plt.close(figure)


def plot_orbits(output_directory: Path, results: dict) -> None:
    """Compare representative orbits for both methods."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        sharex=True,
        sharey=True,
    )
    selected_time_steps = (TIME_STEPS[0], TIME_STEPS[-1])

    for row, (method_name, method_results) in enumerate(results.items()):
        for column, dt in enumerate(selected_time_steps):
            axis = axes[row, column]
            orbit = method_results[dt]["orbit"]
            axis.plot(orbit[:, 0], orbit[:, 1], linewidth=1.0)
            axis.scatter(0.0, 0.0, color="orange", s=50)
            axis.set_title(f"{method_name}, dt = {dt:g}")
            axis.set_xlabel("Relative x [AU]")
            axis.set_ylabel("Relative y [AU]")
            axis.set_aspect("equal")
            axis.grid(True, alpha=0.3)

    figure.suptitle("Orbit comparison")
    figure.tight_layout()
    figure.savefig(output_directory / "orbit_comparison.png", dpi=200)
    plt.close(figure)


def main() -> None:
    output_directory = (
        Path(__file__).resolve().parent / "exercise3_problem1_results"
    )
    output_directory.mkdir(exist_ok=True)

    results = {}

    print("Exercise 3, problem 1")
    print(f"simulation time: {SIMULATION_TIME:g} years")
    print()
    print(
        f"{'method':<20} "
        f"{'dt [year]':>12} "
        f"{'max |dE/E0|':>16} "
        f"{'max |dL/L0|':>16}"
    )

    for method_name, method_module in METHODS.items():
        results[method_name] = {}

        for dt in TIME_STEPS:
            result = simulate(method_module, dt)
            results[method_name][dt] = result
            print(
                f"{method_name:<20} "
                f"{dt:12.6g} "
                f"{result['maximum_energy_error']:16.8e} "
                f"{result['maximum_angular_momentum_error']:16.8e}"
            )

    save_summary(output_directory, results)
    plot_error_history(
        output_directory,
        results,
        "energy_error",
        "total_energy_error.png",
        "Relative total-energy error",
        "Absolute relative error |(E - E0) / E0|",
    )
    plot_error_history(
        output_directory,
        results,
        "angular_momentum_error",
        "angular_momentum_error.png",
        "Relative angular-momentum error",
        "Absolute relative error |(L - L0) / L0|",
    )
    plot_maximum_errors(output_directory, results)
    plot_orbits(output_directory, results)

    print()
    print(f"Results saved in: {output_directory}")


if __name__ == "__main__":
    main()
