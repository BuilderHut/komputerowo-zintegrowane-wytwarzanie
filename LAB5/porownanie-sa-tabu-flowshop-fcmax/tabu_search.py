from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data_io import Instance, parse_data_file
from simulated_annealing import calculate_cmax, format_sequence, instance_name, make_initial_permutation


HistoryRow = Dict[str, Any]
Move = Tuple[int, int]


def swap_move(permutation: np.ndarray, i: int, j: int) -> np.ndarray:
    neighbor = permutation.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def random_swap_moves(
    n: int,
    candidates_per_iteration: int,
    rng: np.random.Generator,
) -> List[Move]:
    """Losuje rozne ruchy typu swap dla pojedynczej iteracji Tabu Search."""
    max_unique = n * (n - 1) // 2
    limit = min(candidates_per_iteration, max_unique)
    moves: set[Move] = set()

    while len(moves) < limit:
        i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
        moves.add((int(i), int(j)))

    return list(moves)


def tabu_search(
    processing_times: np.ndarray,
    initial_permutation: np.ndarray,
    iterations: int = 1000,
    tabu_tenure: int = 15,
    candidates_per_iteration: int = 80,
    seed: int = 123,
) -> Tuple[int, np.ndarray, List[HistoryRow]]:
    """
    Uruchamia Tabu Search dla permutacyjnego problemu przeplywowego F||Cmax.

    W kazdej iteracji oceniana jest losowa probka sasiedztwa swap. Ruch wpisany
    na liste tabu moze zostac wykonany tylko wtedy, gdy spelnia kryterium
    aspiracji, czyli poprawia najlepsze znalezione dotad rozwiazanie.
    """
    if iterations <= 0:
        raise ValueError("Liczba iteracji musi byc dodatnia")
    if tabu_tenure <= 0:
        raise ValueError("Kadencja tabu musi byc dodatnia")
    if candidates_per_iteration <= 0:
        raise ValueError("Liczba kandydatow musi byc dodatnia")

    rng = np.random.default_rng(seed)
    current = np.asarray(initial_permutation, dtype=np.int64).copy()
    current_cmax = calculate_cmax(processing_times, current)
    best = current.copy()
    best_cmax = current_cmax
    tabu_until: Dict[Move, int] = {}
    history: List[HistoryRow] = []
    n = len(current)

    for iteration in range(1, iterations + 1):
        best_candidate: np.ndarray | None = None
        best_candidate_cmax: int | None = None
        best_move: Move | None = None
        best_move_tabu = False

        for move in random_swap_moves(n, candidates_per_iteration, rng):
            candidate = swap_move(current, move[0], move[1])
            candidate_cmax = calculate_cmax(processing_times, candidate)
            is_tabu = tabu_until.get(move, 0) >= iteration
            aspiration = candidate_cmax < best_cmax

            if is_tabu and not aspiration:
                continue

            if best_candidate_cmax is None or candidate_cmax < best_candidate_cmax:
                best_candidate = candidate
                best_candidate_cmax = candidate_cmax
                best_move = move
                best_move_tabu = is_tabu

        if best_candidate is None or best_candidate_cmax is None or best_move is None:
            history.append(
                {
                    "iteration": iteration,
                    "current_cmax": current_cmax,
                    "best_cmax": best_cmax,
                    "candidate_cmax": current_cmax,
                    "move_i": -1,
                    "move_j": -1,
                    "tabu_used": False,
                    "improved_best": False,
                }
            )
            continue

        current = best_candidate
        current_cmax = int(best_candidate_cmax)

        reverse_move = (best_move[0], best_move[1])
        tabu_until[reverse_move] = iteration + tabu_tenure

        improved_best = False
        if current_cmax < best_cmax:
            best = current.copy()
            best_cmax = current_cmax
            improved_best = True

        history.append(
            {
                "iteration": iteration,
                "current_cmax": current_cmax,
                "best_cmax": best_cmax,
                "candidate_cmax": current_cmax,
                "move_i": best_move[0],
                "move_j": best_move[1],
                "tabu_used": best_move_tabu,
                "improved_best": improved_best,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabu Search dla F||Cmax")
    parser.add_argument("data_file", nargs="?", default="data.000.txt", help="plik z instancjami data.xxx")
    parser.add_argument("--instance", default="data.001", help="instancja, np. data.001")
    parser.add_argument("--iterations", type=int, default=1000, help="liczba iteracji")
    parser.add_argument("--tabu-tenure", type=int, default=15, help="czas pozostawania ruchu na liscie tabu")
    parser.add_argument("--candidates", type=int, default=80, help="liczba ocenianych sasiadow w iteracji")
    parser.add_argument("--initial", choices=["neh", "random"], default="neh", help="rozwiazanie startowe")
    parser.add_argument("--seed", type=int, default=123, help="ziarno losowosci")
    parser.add_argument("--history-csv", default="results/ts_history_example.csv", help="plik historii iteracji")
    args = parser.parse_args()

    instances = parse_data_file(args.data_file, require_reference=False)
    instance_id, processing_times, reference_cmax, _ = select_instance(instances, args.instance)

    rng = np.random.default_rng(args.seed)
    initial_cmax, initial_permutation = make_initial_permutation(processing_times, args.initial, rng)

    start = time.perf_counter()
    best_cmax, best_permutation, history = tabu_search(
        processing_times,
        initial_permutation,
        iterations=args.iterations,
        tabu_tenure=args.tabu_tenure,
        candidates_per_iteration=args.candidates,
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
