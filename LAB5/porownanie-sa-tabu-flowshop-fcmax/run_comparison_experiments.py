from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import Instance, parse_data_file
from neh_algorithms import neh_quick_numba
from simulated_annealing import calculate_cmax
from simulated_annealing import instance_name, simulated_annealing
from tabu_search import tabu_search


INITIAL_MODE = "random"

SA_PARAMS = {
    "iterations": 10000,
    "t_start": 1000.0,
    "t_end": 0.01,
    "alpha": 0.995,
    "neighborhood": "swap",
}

TS_PARAMS = {
    "iterations": 500,
    "tabu_tenure": 15,
    "candidates_per_iteration": 40,
}


def parse_instance_range(spec: str, instances: List[Instance]) -> List[Instance]:
    by_name = {instance_name(item[0]): item for item in instances}

    if ":" in spec:
        start_name, end_name = spec.split(":", 1)
        start_id = int(start_name.split(".")[1])
        end_id = int(end_name.split(".")[1])
        return [item for item in instances if start_id <= item[0] <= end_id]

    selected = []
    for name in [part.strip() for part in spec.split(",") if part.strip()]:
        if name not in by_name:
            raise ValueError(f"Nie znaleziono instancji {name}")
        selected.append(by_name[name])
    return selected


def run_sa(instance: Instance, initial_permutation: np.ndarray, initial_cmax: int, seed: int) -> Dict[str, object]:
    instance_id, processing_times, _, _ = instance
    start = time.perf_counter()
    best_cmax, best_permutation, history = simulated_annealing(
        processing_times,
        initial_permutation,
        iterations=int(SA_PARAMS["iterations"]),
        t_start=float(SA_PARAMS["t_start"]),
        t_end=float(SA_PARAMS["t_end"]),
        alpha=float(SA_PARAMS["alpha"]),
        neighborhood=str(SA_PARAMS["neighborhood"]),
        seed=seed,
    )
    elapsed = time.perf_counter() - start

    return {
        "instance": instance_name(instance_id),
        "algorithm": "Simulated Annealing",
        "initial_mode": INITIAL_MODE,
        "initial_cmax": int(initial_cmax),
        "best_cmax": int(best_cmax),
        "time_s": elapsed,
        "iterations": SA_PARAMS["iterations"],
        "objective_evaluations": SA_PARAMS["iterations"],
        "main_parameter": f"T0={SA_PARAMS['t_start']}, alpha={SA_PARAMS['alpha']}",
        "accepted_worse": sum(1 for row in history if row["accepted_worse"]),
        "seed": seed,
        "best_permutation": " ".join(str(int(x) + 1) for x in best_permutation),
    }


def run_ts(instance: Instance, initial_permutation: np.ndarray, initial_cmax: int, seed: int) -> Dict[str, object]:
    instance_id, processing_times, _, _ = instance
    start = time.perf_counter()
    best_cmax, best_permutation, history = tabu_search(
        processing_times,
        initial_permutation,
        iterations=int(TS_PARAMS["iterations"]),
        tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
        candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
        seed=seed,
    )
    elapsed = time.perf_counter() - start

    return {
        "instance": instance_name(instance_id),
        "algorithm": "Tabu Search",
        "initial_mode": INITIAL_MODE,
        "initial_cmax": int(initial_cmax),
        "best_cmax": int(best_cmax),
        "time_s": elapsed,
        "iterations": TS_PARAMS["iterations"],
        "objective_evaluations": TS_PARAMS["iterations"] * TS_PARAMS["candidates_per_iteration"],
        "main_parameter": f"tenure={TS_PARAMS['tabu_tenure']}, cand={TS_PARAMS['candidates_per_iteration']}",
        "accepted_worse": "",
        "seed": seed,
        "best_permutation": " ".join(str(int(x) + 1) for x in best_permutation),
    }


def run_experiments(instances: List[Instance], repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    neh_rows: List[Dict[str, object]] = []

    for instance in instances:
        instance_id, processing_times, reference_cmax, reference_sequence = instance
        neh_start = time.perf_counter()
        if len(reference_sequence) == processing_times.shape[0] and reference_cmax > 0:
            neh_cmax = int(reference_cmax)
            neh_permutation = reference_sequence.copy()
            neh_time = time.perf_counter() - neh_start
        else:
            neh_cmax, neh_permutation = neh_quick_numba(processing_times)
            neh_cmax = int(neh_cmax)
            neh_time = time.perf_counter() - neh_start

        checked_cmax = calculate_cmax(processing_times, neh_permutation)
        if checked_cmax != int(neh_cmax):
            neh_cmax = checked_cmax
        neh_rows.append(
            {
                "instance": instance_name(instance_id),
                "n": processing_times.shape[0],
                "m": processing_times.shape[1],
                "reference_cmax": reference_cmax,
                "neh_cmax": int(neh_cmax),
                "neh_time_s": neh_time,
            }
        )

        for repeat in range(repeats):
            run_seed = seed + repeat + 1000 * int(instance_id)
            if INITIAL_MODE == "random":
                rng = np.random.default_rng(run_seed)
                initial_permutation = rng.permutation(processing_times.shape[0]).astype(np.int64)
            else:
                initial_permutation = neh_permutation.copy()

            initial_cmax = calculate_cmax(processing_times, initial_permutation)
            rows.append(run_sa(instance, initial_permutation.copy(), initial_cmax, run_seed))
            rows.append(run_ts(instance, initial_permutation.copy(), initial_cmax, run_seed))

    runs = pd.DataFrame(rows)
    neh = pd.DataFrame(neh_rows)
    runs = runs.merge(neh, on="instance", how="left")
    runs["difference_vs_neh"] = runs["best_cmax"] - runs["neh_cmax"]
    runs["improvement_vs_neh_percent"] = 100.0 * (runs["neh_cmax"] - runs["best_cmax"]) / runs["neh_cmax"]
    runs["difference_vs_initial"] = runs["best_cmax"] - runs["initial_cmax"]
    runs["improvement_vs_initial_percent"] = 100.0 * (runs["initial_cmax"] - runs["best_cmax"]) / runs["initial_cmax"]

    summary = runs.groupby("algorithm", as_index=False).agg(
        mean_initial_cmax=("initial_cmax", "mean"),
        mean_best_cmax=("best_cmax", "mean"),
        mean_difference_vs_initial=("difference_vs_initial", "mean"),
        mean_improvement_vs_initial_percent=("improvement_vs_initial_percent", "mean"),
        mean_difference_vs_neh=("difference_vs_neh", "mean"),
        mean_improvement_vs_neh_percent=("improvement_vs_neh_percent", "mean"),
        best_improvement_vs_neh_percent=("improvement_vs_neh_percent", "max"),
        worst_improvement_vs_neh_percent=("improvement_vs_neh_percent", "min"),
        mean_time_s=("time_s", "mean"),
        mean_objective_evaluations=("objective_evaluations", "mean"),
        runs=("best_cmax", "count"),
    )

    per_instance = runs.groupby(["instance", "algorithm"], as_index=False).agg(
        n=("n", "first"),
        m=("m", "first"),
        mean_initial_cmax=("initial_cmax", "mean"),
        neh_cmax=("neh_cmax", "first"),
        mean_best_cmax=("best_cmax", "mean"),
        best_cmax=("best_cmax", "min"),
        mean_improvement_vs_initial_percent=("improvement_vs_initial_percent", "mean"),
        mean_improvement_vs_neh_percent=("improvement_vs_neh_percent", "mean"),
        mean_time_s=("time_s", "mean"),
        mean_objective_evaluations=("objective_evaluations", "mean"),
    )
    return runs, summary, per_instance


def save_plots(runs: pd.DataFrame, per_instance: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    pivot = per_instance.pivot(index="instance", columns="algorithm", values="mean_best_cmax")
    neh_values = per_instance.groupby("instance")["neh_cmax"].first().reindex(pivot.index)
    initial_values = per_instance.groupby("instance")["mean_initial_cmax"].first().reindex(pivot.index)
    x = np.arange(len(pivot.index))
    width = 0.21

    plt.figure(figsize=(12, 5))
    plt.bar(x - 1.5 * width, initial_values, width, label="Start losowy")
    plt.bar(x - 0.5 * width, neh_values, width, label="NEH")
    plt.bar(x + 0.5 * width, pivot["Simulated Annealing"], width, label="SA")
    plt.bar(x + 1.5 * width, pivot["Tabu Search"], width, label="Tabu Search")
    plt.title("Porownanie srednich wynikow Cmax")
    plt.xlabel("Instancja")
    plt.ylabel("Cmax")
    plt.xticks(x, pivot.index, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_cmax.png", dpi=150)
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
    plt.title("Rozklad poprawy wzgledem rozwiazania startowego")
    plt.xlabel("Algorytm")
    plt.ylabel("Poprawa [%]")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "improvement_boxplot.png", dpi=150)
    plt.close()

    improvement = per_instance.copy()
    improvement["mean_improvement_cmax"] = improvement["mean_initial_cmax"] - improvement["mean_best_cmax"]
    improvement_pivot = improvement.pivot(index="instance", columns="algorithm", values="mean_improvement_cmax")
    x = np.arange(len(improvement_pivot.index))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, improvement_pivot["Simulated Annealing"], width, label="SA")
    plt.bar(x + width / 2, improvement_pivot["Tabu Search"], width, label="Tabu Search")
    plt.title("Srednia poprawa Cmax wzgledem rozwiazania startowego")
    plt.xlabel("Instancja")
    plt.ylabel("Start Cmax - wynik algorytmu")
    plt.xticks(x, improvement_pivot.index, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "improvement_vs_initial_per_instance.png", dpi=150)
    plt.close()


def save_example_histories(instance: Instance, seed: int, results_dir: Path, plots_dir: Path) -> None:
    instance_id, processing_times, reference_cmax, reference_sequence = instance
    if INITIAL_MODE == "random":
        rng = np.random.default_rng(seed + 1000 * int(instance_id))
        initial_permutation = rng.permutation(processing_times.shape[0]).astype(np.int64)
    elif len(reference_sequence) == processing_times.shape[0] and reference_cmax > 0:
        initial_permutation = reference_sequence.copy()
    else:
        _, initial_permutation = neh_quick_numba(processing_times)

    _, _, sa_history = simulated_annealing(
        processing_times,
        initial_permutation,
        iterations=int(SA_PARAMS["iterations"]),
        t_start=float(SA_PARAMS["t_start"]),
        t_end=float(SA_PARAMS["t_end"]),
        alpha=float(SA_PARAMS["alpha"]),
        neighborhood=str(SA_PARAMS["neighborhood"]),
        seed=seed,
    )
    _, _, ts_history = tabu_search(
        processing_times,
        initial_permutation,
        iterations=int(TS_PARAMS["iterations"]),
        tabu_tenure=int(TS_PARAMS["tabu_tenure"]),
        candidates_per_iteration=int(TS_PARAMS["candidates_per_iteration"]),
        seed=seed,
    )

    sa_df = pd.DataFrame(sa_history)
    ts_df = pd.DataFrame(ts_history)
    sa_df["objective_evaluations"] = sa_df["iteration"]
    ts_df["objective_evaluations"] = ts_df["iteration"] * int(TS_PARAMS["candidates_per_iteration"])
    sa_df.to_csv(results_dir / "sa_history_example.csv", index=False)
    ts_df.to_csv(results_dir / "ts_history_example.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(sa_df["objective_evaluations"], sa_df["best_cmax"], label="SA")
    plt.plot(ts_df["objective_evaluations"], ts_df["best_cmax"], label="Tabu Search")
    if reference_cmax > 0:
        plt.axhline(reference_cmax, color="black", linestyle="--", linewidth=1.2, label="NEH")
    plt.title(f"Przebieg poprawy najlepszego Cmax dla {instance_name(instance_id)}")
    plt.xlabel("Liczba ocen funkcji celu")
    plt.ylabel("Najlepszy Cmax")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "history_best_cmax.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Porownanie Simulated Annealing i Tabu Search dla F||Cmax")
    parser.add_argument("data_file", nargs="?", default="data.000.txt", help="plik z instancjami")
    parser.add_argument("--instances", default="data.111:data.120", help="zakres, np. data.111:data.120")
    parser.add_argument("--repeats", type=int, default=3, help="liczba powtorzen dla instancji")
    parser.add_argument("--seed", type=int, default=123, help="bazowe ziarno losowosci")
    parser.add_argument("--results-dir", default="results", help="katalog wynikow CSV")
    parser.add_argument("--plots-dir", default="plots", help="katalog wykresow")
    args = parser.parse_args()

    all_instances = parse_data_file(args.data_file, require_reference=False)
    instances = parse_instance_range(args.instances, all_instances)
    if not instances:
        raise SystemExit("Nie wybrano zadnych instancji")

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs, summary, per_instance = run_experiments(instances, repeats=args.repeats, seed=args.seed)
    runs.to_csv(results_dir / "comparison_runs.csv", index=False)
    summary.to_csv(results_dir / "comparison_summary.csv", index=False)
    per_instance.to_csv(results_dir / "comparison_per_instance.csv", index=False)
    save_plots(runs, per_instance, plots_dir)
    save_example_histories(instances[0], args.seed, results_dir, plots_dir)

    print(f"Instancje: {args.instances}")
    print(f"Liczba uruchomien metaheurystyk: {len(runs)}")
    print(f"Wyniki zapisano w: {results_dir}")
    print(f"Wykresy zapisano w: {plots_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
