from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import parse_data_file
from simulated_annealing import calculate_cmax, instance_name, simulated_annealing
from tabu_search import tabu_search


INSTANCE_NAME = "data.120"
ITERATION_VALUES = [100, 500, 1000, 2000, 5000, 10000]
HISTORY_ITERATION_VALUES = [100, 1000, 5000, 10000]
REPEATS = 3
BASE_SEED = 2026

SA_PARAMS = {
    "t_start": 1000.0,
    "t_end": 0.01,
    "alpha": 0.995,
    "neighborhood": "swap",
}

TS_PARAMS = {
    "tabu_tenure": 15,
    "candidates_per_iteration": 40,
}


def main() -> None:
    instances = parse_data_file("data.000.txt", require_reference=False)
    instance = next(item for item in instances if instance_name(item[0]) == INSTANCE_NAME)
    instance_id, processing_times, neh_cmax, _ = instance

    rows: List[Dict[str, object]] = []
    for iterations in ITERATION_VALUES:
        for repeat in range(REPEATS):
            seed = BASE_SEED + repeat
            rng = np.random.default_rng(seed)
            initial_permutation = rng.permutation(processing_times.shape[0]).astype(np.int64)
            initial_cmax = calculate_cmax(processing_times, initial_permutation)

            start = time.perf_counter()
            sa_best, _, _ = simulated_annealing(
                processing_times,
                initial_permutation.copy(),
                iterations=iterations,
                t_start=float(SA_PARAMS["t_start"]),
                t_end=float(SA_PARAMS["t_end"]),
                alpha=float(SA_PARAMS["alpha"]),
                neighborhood=str(SA_PARAMS["neighborhood"]),
                seed=seed,
            )
            rows.append(
                {
                    "instance": INSTANCE_NAME,
                    "algorithm": "Simulated Annealing",
                    "iterations": iterations,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "initial_cmax": initial_cmax,
                    "neh_cmax": neh_cmax,
                    "best_cmax": sa_best,
                    "improvement_vs_initial_percent": 100.0 * (initial_cmax - sa_best) / initial_cmax,
                    "objective_evaluations": iterations,
                    "time_s": time.perf_counter() - start,
                }
            )

            start = time.perf_counter()
            ts_best, _, _ = tabu_search(
                processing_times,
                initial_permutation.copy(),
                iterations=iterations,
                tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
                candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
                seed=seed,
            )
            rows.append(
                {
                    "instance": INSTANCE_NAME,
                    "algorithm": "Tabu Search",
                    "iterations": iterations,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "initial_cmax": initial_cmax,
                    "neh_cmax": neh_cmax,
                    "best_cmax": ts_best,
                    "improvement_vs_initial_percent": 100.0 * (initial_cmax - ts_best) / initial_cmax,
                    "objective_evaluations": iterations * int(TS_PARAMS["candidates_per_iteration"]),
                    "time_s": time.perf_counter() - start,
                }
            )

    results_dir = Path("results")
    plots_dir = Path("plots")
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    runs = pd.DataFrame(rows)
    summary = runs.groupby(["iterations", "algorithm"], as_index=False).agg(
        mean_initial_cmax=("initial_cmax", "mean"),
        mean_best_cmax=("best_cmax", "mean"),
        best_cmax=("best_cmax", "min"),
        mean_improvement_vs_initial_percent=("improvement_vs_initial_percent", "mean"),
        mean_objective_evaluations=("objective_evaluations", "mean"),
        mean_time_s=("time_s", "mean"),
        runs=("best_cmax", "count"),
    )
    runs.to_csv(results_dir / "iteration_influence_runs.csv", index=False)
    summary.to_csv(results_dir / "iteration_influence_summary.csv", index=False)

    pivot = summary.pivot(index="iterations", columns="algorithm", values="mean_best_cmax")
    plt.figure(figsize=(9, 5))
    plt.plot(pivot.index, pivot["Simulated Annealing"], marker="o", label="SA")
    plt.plot(pivot.index, pivot["Tabu Search"], marker="o", label="Tabu Search")
    plt.axhline(neh_cmax, color="black", linestyle="--", linewidth=1.2, label="NEH")
    plt.title(f"Wplyw liczby iteracji na wynik dla {INSTANCE_NAME}")
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Sredni najlepszy Cmax")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "iteration_influence_data120.png", dpi=150)
    plt.close()

    history_seed = BASE_SEED
    rng = np.random.default_rng(history_seed)
    history_initial = rng.permutation(processing_times.shape[0]).astype(np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, iterations in zip(axes, HISTORY_ITERATION_VALUES):
        _, _, sa_history = simulated_annealing(
            processing_times,
            history_initial.copy(),
            iterations=iterations,
            t_start=float(SA_PARAMS["t_start"]),
            t_end=float(SA_PARAMS["t_end"]),
            alpha=float(SA_PARAMS["alpha"]),
            neighborhood=str(SA_PARAMS["neighborhood"]),
            seed=history_seed,
        )
        _, _, ts_history = tabu_search(
            processing_times,
            history_initial.copy(),
            iterations=iterations,
            tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
            candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
            seed=history_seed,
        )

        sa_df = pd.DataFrame(sa_history)
        ts_df = pd.DataFrame(ts_history)
        ax.plot(sa_df["iteration"], sa_df["best_cmax"], label="SA", linewidth=1.8)
        ax.plot(ts_df["iteration"], ts_df["best_cmax"], label="Tabu Search", linewidth=1.8)
        ax.axhline(neh_cmax, color="black", linestyle="--", linewidth=1.0, label="NEH")
        ax.set_title(f"{iterations} iteracji")
        ax.set_xlabel("Iteracja")
        ax.set_ylabel("Najlepszy Cmax")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Przebieg najlepszej wartosci Cmax dla {INSTANCE_NAME}", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(plots_dir / "iteration_histories_data120.png", dpi=150)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
