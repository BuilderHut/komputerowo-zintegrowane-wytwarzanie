from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


HistoryRow = Dict[str, float | int | bool]
Edge = Tuple[int, int]


@dataclass(frozen=True)
class TspInstance:
    name: str
    coordinates: np.ndarray
    distances: np.ndarray


def generate_euclidean_instance(name: str, n: int, seed: int) -> TspInstance:
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 1000.0, size=(n, 2))
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return TspInstance(name=name, coordinates=coordinates, distances=distances)


def generate_instances(count: int = 10, n: int = 120, seed: int = 12345) -> List[TspInstance]:
    return [
        generate_euclidean_instance(f"tsp.{i:03d}", n=n, seed=seed + i)
        for i in range(1, count + 1)
    ]


def route_length(distances: np.ndarray, route: np.ndarray) -> float:
    return float(np.sum(distances[route, np.roll(route, -1)]))


def nearest_neighbor_route(distances: np.ndarray, start_city: int = 0) -> np.ndarray:
    n = distances.shape[0]
    unvisited = np.ones(n, dtype=np.bool_)
    route = np.empty(n, dtype=np.int64)
    current = start_city
    route[0] = current
    unvisited[current] = False

    for position in range(1, n):
        candidates = np.where(unvisited)[0]
        next_city = candidates[np.argmin(distances[current, candidates])]
        route[position] = next_city
        unvisited[next_city] = False
        current = int(next_city)

    return route


def random_route(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n).astype(np.int64)


def edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def random_two_opt_move(n: int, rng: np.random.Generator) -> Tuple[int, int]:
    while True:
        i, j = sorted(rng.choice(n, size=2, replace=False).tolist())
        if i == 0 and j == n - 1:
            continue
        return int(i), int(j)


def two_opt_delta(distances: np.ndarray, route: np.ndarray, i: int, j: int) -> float:
    n = len(route)
    a = int(route[i - 1])
    b = int(route[i])
    c = int(route[j])
    d = int(route[(j + 1) % n])
    return float(distances[a, c] + distances[b, d] - distances[a, b] - distances[c, d])


def two_opt_edges(route: np.ndarray, i: int, j: int) -> tuple[tuple[Edge, Edge], tuple[Edge, Edge]]:
    n = len(route)
    a = int(route[i - 1])
    b = int(route[i])
    c = int(route[j])
    d = int(route[(j + 1) % n])
    removed = (edge(a, b), edge(c, d))
    added = (edge(a, c), edge(b, d))
    return removed, added


def apply_two_opt(route: np.ndarray, i: int, j: int) -> None:
    route[i : j + 1] = route[i : j + 1][::-1]


def simulated_annealing_tsp(
    distances: np.ndarray,
    initial_route: np.ndarray,
    iterations: int = 20000,
    t_start: float = 1000.0,
    t_end: float = 0.01,
    alpha: float = 0.9995,
    seed: int = 123,
) -> tuple[float, np.ndarray, List[HistoryRow]]:
    if iterations <= 0:
        raise ValueError("Liczba iteracji musi byc dodatnia")
    if t_start <= 0.0 or t_end <= 0.0:
        raise ValueError("Temperatury musza byc dodatnie")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha musi nalezec do przedzialu (0, 1)")

    rng = np.random.default_rng(seed)
    current = initial_route.copy()
    current_length = route_length(distances, current)
    best = current.copy()
    best_length = current_length
    history: List[HistoryRow] = []

    for iteration in range(1, iterations + 1):
        temperature = max(t_end, t_start * (alpha ** (iteration - 1)))
        i, j = random_two_opt_move(len(current), rng)
        delta = two_opt_delta(distances, current, i, j)

        accepted = False
        accepted_worse = False
        if delta <= 0.0:
            accepted = True
        else:
            probability = math.exp(-delta / max(temperature, 1e-12))
            if rng.random() < probability:
                accepted = True
                accepted_worse = True

        if accepted:
            apply_two_opt(current, i, j)
            current_length += delta
            if current_length < best_length:
                best = current.copy()
                best_length = current_length

        history.append(
            {
                "iteration": iteration,
                "objective_evaluations": iteration,
                "current_length": current_length,
                "best_length": best_length,
                "candidate_length": current_length + delta if not accepted else current_length,
                "delta": delta,
                "temperature": temperature,
                "accepted": accepted,
                "accepted_worse": accepted_worse,
            }
        )

    return float(best_length), best.copy(), history


def tabu_search_tsp(
    distances: np.ndarray,
    initial_route: np.ndarray,
    iterations: int = 500,
    tabu_tenure: int = 20,
    candidates_per_iteration: int = 40,
    seed: int = 123,
) -> tuple[float, np.ndarray, List[HistoryRow]]:
    if iterations <= 0:
        raise ValueError("Liczba iteracji musi byc dodatnia")
    if tabu_tenure <= 0:
        raise ValueError("Kadencja tabu musi byc dodatnia")
    if candidates_per_iteration <= 0:
        raise ValueError("Liczba kandydatow musi byc dodatnia")

    rng = np.random.default_rng(seed)
    current = initial_route.copy()
    current_length = route_length(distances, current)
    best = current.copy()
    best_length = current_length
    tabu_until: Dict[Edge, int] = {}
    history: List[HistoryRow] = []
    n = len(current)

    for iteration in range(1, iterations + 1):
        selected_i = -1
        selected_j = -1
        selected_delta = float("inf")
        selected_removed: tuple[Edge, Edge] | None = None
        selected_tabu = False

        for _ in range(candidates_per_iteration):
            i, j = random_two_opt_move(n, rng)
            delta = two_opt_delta(distances, current, i, j)
            removed_edges, added_edges = two_opt_edges(current, i, j)
            candidate_length = current_length + delta
            is_tabu = any(tabu_until.get(item, 0) >= iteration for item in added_edges)
            aspiration = candidate_length < best_length

            if is_tabu and not aspiration:
                continue

            if delta < selected_delta:
                selected_i = i
                selected_j = j
                selected_delta = delta
                selected_removed = removed_edges
                selected_tabu = is_tabu

        if selected_removed is None:
            history.append(
                {
                    "iteration": iteration,
                    "objective_evaluations": iteration * candidates_per_iteration,
                    "current_length": current_length,
                    "best_length": best_length,
                    "candidate_length": current_length,
                    "delta": 0.0,
                    "tabu_used": False,
                    "improved_best": False,
                }
            )
            continue

        apply_two_opt(current, selected_i, selected_j)
        current_length += selected_delta
        for item in selected_removed:
            tabu_until[item] = iteration + tabu_tenure

        improved_best = False
        if current_length < best_length:
            best = current.copy()
            best_length = current_length
            improved_best = True

        history.append(
            {
                "iteration": iteration,
                "objective_evaluations": iteration * candidates_per_iteration,
                "current_length": current_length,
                "best_length": best_length,
                "candidate_length": current_length,
                "delta": selected_delta,
                "tabu_used": selected_tabu,
                "improved_best": improved_best,
            }
        )

    return float(best_length), best.copy(), history
