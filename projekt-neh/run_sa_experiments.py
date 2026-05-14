from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import Instance, parse_data_file
from neh_algorithms import neh_quick_numba
from simulated_annealing import instance_name, make_initial_permutation, simulated_annealing


BASE_PARAMS = {
    "iterations": 10000,
    "t_start": 1000.0,
    "t_end": 0.01,
    "alpha": 0.995,
    "initial": "neh",
    "neighborhood": "swap",
}


def parse_instance_range(spec: str, instances: List[Instance]) -> List[Instance]:
    """Wybiera instancje według zapisu data.001:data.020 albo listy po przecinku."""
    by_name = {instance_name(item[0]): item for item in instances}

    if ":" in spec:
        start_name, end_name = spec.split(":", 1)
        start_id = int(start_name.split(".")[1])
        end_id = int(end_name.split(".")[1])
        return [item for item in instances if start_id <= item[0] <= end_id]

    names = [part.strip() for part in spec.split(",") if part.strip()]
    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Nie znaleziono instancji {name}")
        selected.append(by_name[name])
    return selected


def experiment_variants() -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []

    for value in [1000, 5000, 10000, 50000]:
        params = dict(BASE_PARAMS)
        params["iterations"] = value
        variants.append({"experiment": "iterations", "variant": str(value), **params})

    for value in [10.0, 100.0, 1000.0, 10000.0]:
        params = dict(BASE_PARAMS)
        params["t_start"] = value
        variants.append({"experiment": "t_start", "variant": str(int(value)), **params})

    for value in [0.90, 0.95, 0.99, 0.995]:
        params = dict(BASE_PARAMS)
        params["alpha"] = value
        variants.append({"experiment": "alpha", "variant": str(value), **params})

    for value in ["neh", "random"]:
        params = dict(BASE_PARAMS)
        params["initial"] = value
        variants.append({"experiment": "initial", "variant": value, **params})

    for value in ["swap", "insert"]:
        params = dict(BASE_PARAMS)
        params["neighborhood"] = value
        variants.append({"experiment": "neighborhood", "variant": value, **params})

    return variants


def run_single_sa(instance: Instance, params: Dict[str, object], seed: int) -> Dict[str, object]:
    instance_id, processing_times, reference_cmax, _ = instance

    neh_start_time = time.perf_counter()
    neh_cmax, neh_permutation = neh_quick_numba(processing_times)
    neh_time = time.perf_counter() - neh_start_time

    rng = np.random.default_rng(seed)
    if params["initial"] == "neh":
        initial_cmax = int(neh_cmax)
        initial_permutation = neh_permutation.copy()
    else:
        initial_cmax, initial_permutation = make_initial_permutation(processing_times, "random", rng)

    sa_start_time = time.perf_counter()
    best_cmax, best_permutation, history = simulated_annealing(
        processing_times,
        initial_permutation,
        iterations=int(params["iterations"]),
        t_start=float(params["t_start"]),
        t_end=float(params["t_end"]),
        alpha=float(params["alpha"]),
        neighborhood=str(params["neighborhood"]),
        seed=seed,
    )
    sa_time = time.perf_counter() - sa_start_time

    difference = best_cmax - int(neh_cmax)
    improvement_percent = 100.0 * (int(neh_cmax) - best_cmax) / int(neh_cmax)
    accepted_worse_count = sum(1 for row in history if row["accepted_worse"])

    return {
        "instance": instance_name(instance_id),
        "n": processing_times.shape[0],
        "m": processing_times.shape[1],
        "reference_cmax": reference_cmax,
        "neh_cmax": int(neh_cmax),
        "initial_cmax": int(initial_cmax),
        "sa_best_cmax": int(best_cmax),
        "difference": int(difference),
        "improvement_percent": improvement_percent,
        "sa_time": sa_time,
        "neh_time": neh_time,
        "accepted_worse_count": accepted_worse_count,
        "seed": seed,
        "best_permutation": " ".join(str(int(x) + 1) for x in best_permutation),
        **params,
    }


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(["experiment", "variant"], as_index=False)
    return grouped.agg(
        mean_sa_best_cmax=("sa_best_cmax", "mean"),
        mean_improvement_percent=("improvement_percent", "mean"),
        min_improvement_percent=("improvement_percent", "min"),
        max_improvement_percent=("improvement_percent", "max"),
        mean_sa_time=("sa_time", "mean"),
        runs=("sa_best_cmax", "count"),
    )


def save_example_history_plot(history: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["iteration"], history["current_cmax"], label="Aktualny Cmax", alpha=0.75)
    plt.plot(history["iteration"], history["best_cmax"], label="Najlepszy Cmax", linewidth=2)
    plt.title("Przebieg symulowanego wyżarzania")
    plt.xlabel("Iteracja")
    plt.ylabel("Cmax")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_parameter_plot(summary: pd.DataFrame, experiment: str, title: str, path: Path) -> None:
    subset = summary[summary["experiment"] == experiment].copy()
    if subset.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(subset["variant"], subset["mean_improvement_percent"])
    plt.title(title)
    plt.xlabel("Wariant parametru")
    plt.ylabel("Średnia poprawa względem NEH [%]")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison_plot(comparison: pd.DataFrame, path: Path) -> None:
    subset = comparison.head(20).copy()
    x = np.arange(len(subset))
    width = 0.38

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, subset["neh_cmax"], width, label="NEH")
    plt.bar(x + width / 2, subset["sa_best_cmax"], width, label="SA")
    plt.title("Porównanie NEH i symulowanego wyżarzania")
    plt.xlabel("Instancja")
    plt.ylabel("Cmax")
    plt.xticks(x, subset["instance"], rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    baseline = runs[(runs["experiment"] == "initial") & (runs["variant"] == "neh")].copy()
    if baseline.empty:
        baseline = runs.copy()

    best_per_instance = baseline.sort_values("sa_best_cmax").groupby("instance", as_index=False).first()
    return best_per_instance[
        [
            "instance",
            "n",
            "m",
            "neh_cmax",
            "sa_best_cmax",
            "difference",
            "improvement_percent",
            "sa_time",
            "neh_time",
            "iterations",
            "t_start",
            "t_end",
            "alpha",
            "initial",
            "neighborhood",
            "seed",
        ]
    ]


def save_plots(summary: pd.DataFrame, comparison: pd.DataFrame, example_history: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_example_history_plot(example_history, plots_dir / "cmax_current_vs_best_example.png")
    save_parameter_plot(summary, "iterations", "Wpływ liczby iteracji na wynik SA", plots_dir / "influence_iterations.png")
    save_parameter_plot(summary, "t_start", "Wpływ temperatury początkowej na wynik SA", plots_dir / "influence_temperature.png")
    save_parameter_plot(summary, "alpha", "Wpływ współczynnika chłodzenia alpha", plots_dir / "influence_alpha.png")
    save_parameter_plot(summary, "initial", "Wpływ rozwiązania początkowego na wynik SA", plots_dir / "influence_initial.png")
    save_parameter_plot(summary, "neighborhood", "Wpływ sąsiedztwa na wynik SA", plots_dir / "influence_neighborhood.png")
    save_comparison_plot(comparison, plots_dir / "comparison_neh_sa.png")


def run_experiments(instances: List[Instance], repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []

    for variant in experiment_variants():
        for instance in instances:
            for repeat in range(repeats):
                run_seed = seed + repeat + 1000 * int(instance[0])
                rows.append(run_single_sa(instance, variant, run_seed))

    runs = pd.DataFrame(rows)

    first_instance = instances[0]
    _, processing_times, _, _ = first_instance
    _, initial_permutation = neh_quick_numba(processing_times)
    _, _, history = simulated_annealing(
        processing_times,
        initial_permutation,
        iterations=int(BASE_PARAMS["iterations"]),
        t_start=float(BASE_PARAMS["t_start"]),
        t_end=float(BASE_PARAMS["t_end"]),
        alpha=float(BASE_PARAMS["alpha"]),
        neighborhood=str(BASE_PARAMS["neighborhood"]),
        seed=seed,
    )
    return runs, pd.DataFrame(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Badania symulowanego wyżarzania dla F||Cmax")
    parser.add_argument("data_file", nargs="?", default="data.000.txt", help="plik z instancjami")
    parser.add_argument("--instances", default="data.001:data.020", help="zakres, np. data.001:data.020")
    parser.add_argument("--repeats", type=int, default=5, help="liczba powtórzeń dla wariantu")
    parser.add_argument("--seed", type=int, default=123, help="bazowe ziarno losowości")
    parser.add_argument("--results-dir", default="results", help="katalog wyników CSV")
    parser.add_argument("--plots-dir", default="plots", help="katalog wykresów")
    args = parser.parse_args()

    all_instances = parse_data_file(args.data_file, require_reference=False)
    instances = parse_instance_range(args.instances, all_instances)
    if not instances:
        raise SystemExit("Nie wybrano żadnych instancji")

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    runs, example_history = run_experiments(instances, repeats=args.repeats, seed=args.seed)
    summary = summarize_runs(runs)
    comparison = build_comparison(runs)

    runs.to_csv(results_dir / "sa_runs.csv", index=False)
    summary.to_csv(results_dir / "sa_summary.csv", index=False)
    comparison.to_csv(results_dir / "comparison_neh_sa.csv", index=False)
    example_history.to_csv(results_dir / "sa_history_example.csv", index=False)
    save_plots(summary, comparison, example_history, plots_dir)

    print(f"Instancje: {args.instances}")
    print(f"Liczba uruchomień SA: {len(runs)}")
    print(f"Wyniki zapisano w: {results_dir}")
    print(f"Wykresy zapisano w: {plots_dir}")


if __name__ == "__main__":
    main()
