from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsp_algorithms import (
    generate_instances,
    nearest_neighbor_route,
    random_route,
    route_length,
    simulated_annealing_tsp,
    tabu_search_tsp,
)


INSTANCE_INDEX = 10
CITIES = 120
BASE_SEED = 2026
ITERATION_VALUES = [100, 500, 1000, 2000, 5000, 10000]
HISTORY_ITERATION_VALUES = [100, 1000, 5000, 10000]
REPEATS = 3

SA_PARAMS = {
    "t_start": 1000.0,
    "t_end": 0.01,
    "alpha": 0.9995,
}

TS_PARAMS = {
    "tabu_tenure": 20,
    "candidates_per_iteration": 40,
}


def main() -> None:
    instance = generate_instances(count=INSTANCE_INDEX, n=CITIES, seed=BASE_SEED)[-1]
    nn_route = nearest_neighbor_route(instance.distances)
    nn_length = route_length(instance.distances, nn_route)
    rows: List[Dict[str, object]] = []

    for iterations in ITERATION_VALUES:
        for repeat in range(REPEATS):
            seed = BASE_SEED + repeat
            initial_route = random_route(CITIES, seed)
            initial_length = route_length(instance.distances, initial_route)

            sa_best, _, _ = simulated_annealing_tsp(
                instance.distances,
                initial_route.copy(),
                iterations=iterations,
                t_start=float(SA_PARAMS["t_start"]),
                t_end=float(SA_PARAMS["t_end"]),
                alpha=float(SA_PARAMS["alpha"]),
                seed=seed,
            )
            rows.append(
                {
                    "instance": instance.name,
                    "algorithm": "Simulated Annealing",
                    "iterations": iterations,
                    "repeat": repeat + 1,
                    "initial_length": initial_length,
                    "nearest_neighbor_length": nn_length,
                    "best_length": sa_best,
                    "improvement_vs_initial_percent": 100.0 * (initial_length - sa_best) / initial_length,
                    "objective_evaluations": iterations,
                    "seed": seed,
                }
            )

            ts_best, _, _ = tabu_search_tsp(
                instance.distances,
                initial_route.copy(),
                iterations=iterations,
                tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
                candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
                seed=seed,
            )
            rows.append(
                {
                    "instance": instance.name,
                    "algorithm": "Tabu Search",
                    "iterations": iterations,
                    "repeat": repeat + 1,
                    "initial_length": initial_length,
                    "nearest_neighbor_length": nn_length,
                    "best_length": ts_best,
                    "improvement_vs_initial_percent": 100.0 * (initial_length - ts_best) / initial_length,
                    "objective_evaluations": iterations * int(TS_PARAMS["candidates_per_iteration"]),
                    "seed": seed,
                }
            )

    results_dir = Path("results")
    plots_dir = Path("plots")
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    runs = pd.DataFrame(rows)
    summary = runs.groupby(["iterations", "algorithm"], as_index=False).agg(
        mean_initial_length=("initial_length", "mean"),
        mean_best_length=("best_length", "mean"),
        best_length=("best_length", "min"),
        mean_improvement_vs_initial_percent=("improvement_vs_initial_percent", "mean"),
        mean_objective_evaluations=("objective_evaluations", "mean"),
        runs=("best_length", "count"),
    )
    runs.to_csv(results_dir / "iteration_influence_runs.csv", index=False)
    summary.to_csv(results_dir / "iteration_influence_summary.csv", index=False)

    pivot = summary.pivot(index="iterations", columns="algorithm", values="mean_best_length")
    plt.figure(figsize=(9, 5))
    plt.plot(pivot.index, pivot["Simulated Annealing"], marker="o", label="SA")
    plt.plot(pivot.index, pivot["Tabu Search"], marker="o", label="Tabu Search")
    plt.axhline(nn_length, color="black", linestyle="--", linewidth=1.2, label="Najblizszy sasiad")
    plt.title(f"Wplyw liczby iteracji na wynik dla {instance.name}")
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Srednia najlepsza dlugosc trasy")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "iteration_influence_tsp010.png", dpi=150)
    plt.close()

    history_seed = BASE_SEED
    history_initial = random_route(CITIES, history_seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, iterations in zip(axes, HISTORY_ITERATION_VALUES):
        _, _, sa_history = simulated_annealing_tsp(
            instance.distances,
            history_initial.copy(),
            iterations=iterations,
            t_start=float(SA_PARAMS["t_start"]),
            t_end=float(SA_PARAMS["t_end"]),
            alpha=float(SA_PARAMS["alpha"]),
            seed=history_seed,
        )
        _, _, ts_history = tabu_search_tsp(
            instance.distances,
            history_initial.copy(),
            iterations=iterations,
            tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
            candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
            seed=history_seed,
        )
        sa_df = pd.DataFrame(sa_history)
        ts_df = pd.DataFrame(ts_history)
        ax.plot(sa_df["iteration"], sa_df["best_length"], label="SA", linewidth=1.8)
        ax.plot(ts_df["iteration"], ts_df["best_length"], label="Tabu Search", linewidth=1.8)
        ax.axhline(nn_length, color="black", linestyle="--", linewidth=1.0, label="Najblizszy sasiad")
        ax.set_title(f"{iterations} iteracji")
        ax.set_xlabel("Iteracja")
        ax.set_ylabel("Najlepsza dlugosc")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Przebieg najlepszej dlugosci trasy dla {instance.name}", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(plots_dir / "iteration_histories_tsp010.png", dpi=150)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
