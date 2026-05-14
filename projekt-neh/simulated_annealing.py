from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data_io import Instance, parse_data_file
from neh_algorithms import cmax_numba, neh_quick_numba


HistoryRow = Dict[str, Any]


def instance_name(instance_id: int) -> str:
    return f"data.{instance_id:03d}"


def calculate_cmax(processing_times: np.ndarray, permutation: np.ndarray) -> int:
    """Liczy Cmax dla dowolnej permutacji zadań."""
    sequence = np.asarray(permutation, dtype=np.int64)
    return int(cmax_numba(processing_times, sequence, len(sequence)))


def generate_neighbor_swap(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tworzy sąsiada przez zamianę dwóch różnych pozycji."""
    neighbor = permutation.copy()
    n = len(neighbor)
    i, j = rng.choice(n, size=2, replace=False)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def generate_neighbor_insert(permutation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tworzy sąsiada przez wyjęcie zadania i wstawienie go w inną pozycję."""
    n = len(permutation)
    source, target = rng.choice(n, size=2, replace=False)

    values = list(int(x) for x in permutation)
    job = values.pop(int(source))
    values.insert(int(target), job)
    return np.array(values, dtype=np.int64)


def make_initial_permutation(
    processing_times: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[int, np.ndarray]:
    """Zwraca permutację startową NEH albo losową."""
    if mode == "neh":
        cmax, permutation = neh_quick_numba(processing_times)
        return int(cmax), permutation.copy()

    if mode == "random":
        permutation = rng.permutation(processing_times.shape[0]).astype(np.int64)
        return calculate_cmax(processing_times, permutation), permutation

    raise ValueError(f"Nieznany typ rozwiązania początkowego: {mode}")


def cooling_temperature(
    iteration: int,
    iterations: int,
    t_start: float,
    t_end: float,
    alpha: float | None,
) -> float:
    """Wyznacza temperaturę dla danej iteracji."""
    if alpha is not None:
        return max(t_end, t_start * (alpha ** iteration))

    if iterations <= 1:
        return t_end

    ratio = iteration / (iterations - 1)
    return t_start * ((t_end / t_start) ** ratio)


def simulated_annealing(
    processing_times: np.ndarray,
    initial_permutation: np.ndarray,
    iterations: int = 10000,
    t_start: float = 1000.0,
    t_end: float = 0.01,
    alpha: float | None = None,
    neighborhood: str = "swap",
    seed: int = 123,
) -> Tuple[int, np.ndarray, List[HistoryRow]]:
    """
    Uruchamia symulowane wyżarzanie dla permutacyjnego problemu przepływowego.

    Gorsze rozwiązanie może zostać przyjęte z prawdopodobieństwem zależnym od
    temperatury. Na początku algorytm łatwiej wychodzi z minimów lokalnych,
    a pod koniec zachowuje się coraz bardziej zachłannie.
    """
    if iterations <= 0:
        raise ValueError("Liczba iteracji musi być dodatnia")
    if t_start <= 0 or t_end <= 0:
        raise ValueError("Temperatury muszą być dodatnie")
    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError("alpha musi należeć do przedziału (0, 1)")
    if neighborhood not in {"swap", "insert"}:
        raise ValueError("neighborhood musi mieć wartość 'swap' albo 'insert'")

    rng = np.random.default_rng(seed)
    current = np.asarray(initial_permutation, dtype=np.int64).copy()
    current_cmax = calculate_cmax(processing_times, current)
    best = current.copy()
    best_cmax = current_cmax
    history: List[HistoryRow] = []

    neighbor_fn = generate_neighbor_swap if neighborhood == "swap" else generate_neighbor_insert

    for iteration in range(1, iterations + 1):
        temperature = cooling_temperature(iteration - 1, iterations, t_start, t_end, alpha)
        candidate = neighbor_fn(current, rng)
        candidate_cmax = calculate_cmax(processing_times, candidate)
        delta = candidate_cmax - current_cmax

        accepted = False
        accepted_worse = False

        if delta <= 0:
            accepted = True
        else:
            # Im wyższa temperatura, tym większa szansa na zaakceptowanie
            # pogorszenia. To pozwala przeskakiwać przez lokalne minima.
            probability = math.exp(-delta / max(temperature, 1e-12))
            if rng.random() < probability:
                accepted = True
                accepted_worse = True

        if accepted:
            current = candidate
            current_cmax = candidate_cmax

            if current_cmax < best_cmax:
                best = current.copy()
                best_cmax = current_cmax

        history.append(
            {
                "iteration": iteration,
                "temperature": temperature,
                "current_cmax": current_cmax,
                "best_cmax": best_cmax,
                "candidate_cmax": candidate_cmax,
                "delta": delta,
                "accepted": accepted,
                "accepted_worse": accepted_worse,
            }
        )

    return int(best_cmax), best.copy(), history


def select_instance(instances: List[Instance], name: str) -> Instance:
    for item in instances:
        if instance_name(item[0]) == name:
            return item
    raise ValueError(f"Nie znaleziono instancji {name}")


def save_history(history: List[HistoryRow], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def format_sequence(sequence: np.ndarray) -> str:
    return " ".join(str(int(x) + 1) for x in sequence)


def main() -> None:
    parser = argparse.ArgumentParser(description="Symulowane wyżarzanie dla F||Cmax")
    parser.add_argument("data_file", nargs="?", default="data.000.txt", help="plik z instancjami data.xxx")
    parser.add_argument("--instance", default="data.001", help="instancja, np. data.001")
    parser.add_argument("--iterations", type=int, default=10000, help="liczba iteracji")
    parser.add_argument("--t-start", type=float, default=1000.0, help="temperatura początkowa")
    parser.add_argument("--t-end", type=float, default=0.01, help="temperatura końcowa")
    parser.add_argument("--alpha", type=float, default=0.995, help="współczynnik chłodzenia")
    parser.add_argument("--initial", choices=["neh", "random"], default="neh", help="rozwiązanie startowe")
    parser.add_argument("--neighborhood", choices=["swap", "insert"], default="swap", help="typ sąsiedztwa")
    parser.add_argument("--seed", type=int, default=123, help="ziarno losowości")
    parser.add_argument("--history-csv", default="results/sa_history_example.csv", help="plik historii iteracji")
    args = parser.parse_args()

    instances = parse_data_file(args.data_file, require_reference=False)
    instance_id, processing_times, reference_cmax, _ = select_instance(instances, args.instance)

    rng = np.random.default_rng(args.seed)
    initial_cmax, initial_permutation = make_initial_permutation(processing_times, args.initial, rng)

    start = time.perf_counter()
    best_cmax, best_permutation, history = simulated_annealing(
        processing_times,
        initial_permutation,
        iterations=args.iterations,
        t_start=args.t_start,
        t_end=args.t_end,
        alpha=args.alpha,
        neighborhood=args.neighborhood,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    save_history(history, args.history_csv)

    print(f"instance: {instance_name(instance_id)}")
    print(f"reference_neh_cmax: {reference_cmax}")
    print(f"initial: {args.initial}")
    print(f"initial_cmax: {initial_cmax}")
    print(f"best_cmax: {best_cmax}")
    print(f"improvement_vs_initial: {initial_cmax - best_cmax}")
    print(f"time_s: {elapsed:.6f}")
    print(f"best_permutation: {format_sequence(best_permutation)}")
    print(f"history_csv: {args.history_csv}")


if __name__ == "__main__":
    main()
