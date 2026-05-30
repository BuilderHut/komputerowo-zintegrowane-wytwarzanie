from __future__ import annotations

import time
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


INSTANCES = 10
CITIES = 120
REPEATS = 5
BASE_SEED = 2026

SA_PARAMS = {
    "iterations": 20000,
    "t_start": 1000.0,
    "t_end": 0.01,
    "alpha": 0.9995,
}

TS_PARAMS = {
    "iterations": 500,
    "tabu_tenure": 20,
    "candidates_per_iteration": 40,
}


def run_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instances = generate_instances(count=INSTANCES, n=CITIES, seed=BASE_SEED)
    rows: List[Dict[str, object]] = []

    for instance_index, instance in enumerate(instances, start=1):
        nn_route = nearest_neighbor_route(instance.distances)
        nn_length = route_length(instance.distances, nn_route)

        for repeat in range(REPEATS):
            seed = BASE_SEED + 1000 * instance_index + repeat
            initial_route = random_route(CITIES, seed)
            initial_length = route_length(instance.distances, initial_route)

            start = time.perf_counter()
            sa_best, _, sa_history = simulated_annealing_tsp(
                instance.distances,
                initial_route.copy(),
                iterations=int(SA_PARAMS["iterations"]),
                t_start=float(SA_PARAMS["t_start"]),
                t_end=float(SA_PARAMS["t_end"]),
                alpha=float(SA_PARAMS["alpha"]),
                seed=seed,
            )
            rows.append(
                {
                    "instance": instance.name,
                    "algorithm": "Simulated Annealing",
                    "cities": CITIES,
                    "initial_length": initial_length,
                    "nearest_neighbor_length": nn_length,
                    "best_length": sa_best,
                    "improvement_vs_initial_percent": 100.0 * (initial_length - sa_best) / initial_length,
                    "difference_vs_nearest_neighbor": sa_best - nn_length,
                    "objective_evaluations": SA_PARAMS["iterations"],
                    "time_s": time.perf_counter() - start,
                    "seed": seed,
                    "accepted_worse": sum(1 for row in sa_history if row["accepted_worse"]),
                }
            )

            start = time.perf_counter()
            ts_best, _, _ = tabu_search_tsp(
                instance.distances,
                initial_route.copy(),
                iterations=int(TS_PARAMS["iterations"]),
                tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
                candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
                seed=seed,
            )
            rows.append(
                {
                    "instance": instance.name,
                    "algorithm": "Tabu Search",
                    "cities": CITIES,
                    "initial_length": initial_length,
                    "nearest_neighbor_length": nn_length,
                    "best_length": ts_best,
                    "improvement_vs_initial_percent": 100.0 * (initial_length - ts_best) / initial_length,
                    "difference_vs_nearest_neighbor": ts_best - nn_length,
                    "objective_evaluations": TS_PARAMS["iterations"] * TS_PARAMS["candidates_per_iteration"],
                    "time_s": time.perf_counter() - start,
                    "seed": seed,
                    "accepted_worse": "",
                }
            )

    runs = pd.DataFrame(rows)
    summary = runs.groupby("algorithm", as_index=False).agg(
        mean_initial_length=("initial_length", "mean"),
        mean_best_length=("best_length", "mean"),
        mean_improvement_vs_initial_percent=("improvement_vs_initial_percent", "mean"),
        mean_difference_vs_nearest_neighbor=("difference_vs_nearest_neighbor", "mean"),
        mean_time_s=("time_s", "mean"),
        mean_objective_evaluations=("objective_evaluations", "mean"),
        runs=("best_length", "count"),
    )
    per_instance = runs.groupby(["instance", "algorithm"], as_index=False).agg(
        cities=("cities", "first"),
        mean_initial_length=("initial_length", "mean"),
        nearest_neighbor_length=("nearest_neighbor_length", "first"),
        mean_best_length=("best_length", "mean"),
        best_length=("best_length", "min"),
        mean_improvement_vs_initial_percent=("improvement_vs_initial_percent", "mean"),
        mean_time_s=("time_s", "mean"),
    )
    return runs, summary, per_instance


def save_example_history(plots_dir: Path, results_dir: Path) -> None:
    instance = generate_instances(count=1, n=CITIES, seed=BASE_SEED)[0]
    nn_route = nearest_neighbor_route(instance.distances)
    nn_length = route_length(instance.distances, nn_route)
    seed = BASE_SEED + 1000
    initial_route = random_route(CITIES, seed)

    _, _, sa_history = simulated_annealing_tsp(
        instance.distances,
        initial_route.copy(),
        iterations=int(SA_PARAMS["iterations"]),
        t_start=float(SA_PARAMS["t_start"]),
        t_end=float(SA_PARAMS["t_end"]),
        alpha=float(SA_PARAMS["alpha"]),
        seed=seed,
    )
    _, _, ts_history = tabu_search_tsp(
        instance.distances,
        initial_route.copy(),
        iterations=int(TS_PARAMS["iterations"]),
        tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
        candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
        seed=seed,
    )

    sa_df = pd.DataFrame(sa_history)
    ts_df = pd.DataFrame(ts_history)
    sa_df.to_csv(results_dir / "sa_history_example.csv", index=False)
    ts_df.to_csv(results_dir / "ts_history_example.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(sa_df["objective_evaluations"], sa_df["best_length"], label="SA")
    plt.plot(ts_df["objective_evaluations"], ts_df["best_length"], label="Tabu Search")
    plt.axhline(nn_length, color="black", linestyle="--", linewidth=1.2, label="Najblizszy sasiad")
    plt.title(f"Przebieg poprawy najlepszej trasy dla {instance.name}")
    plt.xlabel("Liczba ocen funkcji celu")
    plt.ylabel("Najlepsza dlugosc trasy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "history_best_length.png", dpi=150)
    plt.close()


def save_plots(runs: pd.DataFrame, per_instance: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    pivot = per_instance.pivot(index="instance", columns="algorithm", values="mean_best_length")
    initial_values = per_instance.groupby("instance")["mean_initial_length"].first().reindex(pivot.index)
    nn_values = per_instance.groupby("instance")["nearest_neighbor_length"].first().reindex(pivot.index)
    x = np.arange(len(pivot.index))
    width = 0.21

    plt.figure(figsize=(12, 5))
    plt.bar(x - 1.5 * width, initial_values, width, label="Start losowy")
    plt.bar(x - 0.5 * width, nn_values, width, label="Najblizszy sasiad")
    plt.bar(x + 0.5 * width, pivot["Simulated Annealing"], width, label="SA")
    plt.bar(x + 1.5 * width, pivot["Tabu Search"], width, label="Tabu Search")
    plt.title("Porownanie sredniej dlugosci trasy")
    plt.xlabel("Instancja")
    plt.ylabel("Dlugosc trasy")
    plt.xticks(x, pivot.index, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_length.png", dpi=150)
    plt.close()

    improvement = per_instance.copy()
    improvement["improvement_length"] = improvement["mean_initial_length"] - improvement["mean_best_length"]
    improvement_pivot = improvement.pivot(index="instance", columns="algorithm", values="improvement_length")
    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, improvement_pivot["Simulated Annealing"], 0.35, label="SA")
    plt.bar(x + width / 2, improvement_pivot["Tabu Search"], 0.35, label="Tabu Search")
    plt.title("Poprawa dlugosci trasy wzgledem startu losowego")
    plt.xlabel("Instancja")
    plt.ylabel("Start - wynik")
    plt.xticks(x, improvement_pivot.index, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "improvement_vs_initial.png", dpi=150)
    plt.close()

    time_pivot = per_instance.pivot(index="instance", columns="algorithm", values="mean_time_s")
    plt.figure(figsize=(10, 5))
    plt.plot(time_pivot.index, time_pivot["Simulated Annealing"], marker="o", label="SA")
    plt.plot(time_pivot.index, time_pivot["Tabu Search"], marker="o", label="Tabu Search")
    plt.title("Porownanie czasu wykonania")
    plt.xlabel("Instancja")
    plt.ylabel("Czas [s]")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    runs.boxplot(column="improvement_vs_initial_percent", by="algorithm")
    plt.suptitle("")
    plt.title("Rozklad poprawy wzgledem startu losowego")
    plt.xlabel("Algorytm")
    plt.ylabel("Poprawa [%]")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "improvement_boxplot.png", dpi=150)
    plt.close()


def main() -> None:
    results_dir = Path("results")
    plots_dir = Path("plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs, summary, per_instance = run_experiments()
    runs.to_csv(results_dir / "comparison_runs.csv", index=False)
    summary.to_csv(results_dir / "comparison_summary.csv", index=False)
    per_instance.to_csv(results_dir / "comparison_per_instance.csv", index=False)
    save_plots(runs, per_instance, plots_dir)
    save_example_history(plots_dir, results_dir)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
