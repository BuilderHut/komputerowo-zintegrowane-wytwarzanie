from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List

import numpy as np

from data_io import Instance
from neh_algorithms import neh_naive_numba, neh_quick_numba


def format_sequence(sequence: np.ndarray) -> str:
    """Zamienia kolejność 0-based na tekst 1-based."""
    return " ".join(str(int(x) + 1) for x in sequence)


def run_benchmark(instances: List[Instance], quick_only: bool, csv_path: str | Path) -> None:
    # Rozgrzewka numby: kompilacja JIT nie jest liczona do pomiaru algorytmu.
    neh_quick_numba(instances[0][1])
    if not quick_only:
        neh_naive_numba(instances[0][1])

    rows = []
    quick_total = 0.0
    naive_total = 0.0
    quick_max = 0.0
    naive_max = 0.0
    wrong = 0

    print("instance,n,m,ref,quick,quick_time_s,naive,naive_time_s,ok")

    for instance_id, p, reference_cmax, reference_sequence in instances:
        n, m = p.shape

        start = time.perf_counter()
        quick_cmax, quick_sequence = neh_quick_numba(p)
        quick_time = time.perf_counter() - start
        quick_total += quick_time
        quick_max = max(quick_max, quick_time)

        if quick_only:
            naive_cmax = -1
            naive_time = 0.0
            naive_sequence = reference_sequence
        else:
            start = time.perf_counter()
            naive_cmax, naive_sequence = neh_naive_numba(p)
            naive_time = time.perf_counter() - start
            naive_total += naive_time
            naive_max = max(naive_max, naive_time)

        quick_ok = quick_cmax == reference_cmax and np.array_equal(quick_sequence, reference_sequence)
        naive_ok = quick_only or (
            naive_cmax == reference_cmax and np.array_equal(naive_sequence, reference_sequence)
        )
        ok = quick_ok and naive_ok
        if not ok:
            wrong += 1

        naive_cmax_text = "" if quick_only else str(int(naive_cmax))
        naive_time_text = "" if quick_only else f"{naive_time:.6f}"
        print(
            f"data.{instance_id:03d},{n},{m},{reference_cmax},"
            f"{quick_cmax},{quick_time:.6f},"
            f"{naive_cmax_text},{naive_time_text},"
            f"{ok}"
        )

        rows.append(
            {
                "instance": f"data.{instance_id:03d}",
                "n": n,
                "m": m,
                "reference_cmax": reference_cmax,
                "quick_cmax": int(quick_cmax),
                "quick_time_s": f"{quick_time:.9f}",
                "quick_sequence": format_sequence(quick_sequence),
                "naive_cmax": "" if quick_only else int(naive_cmax),
                "naive_time_s": "" if quick_only else f"{naive_time:.9f}",
                "naive_sequence": "" if quick_only else format_sequence(naive_sequence),
                "ok": ok,
            }
        )

    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    speedup = "-"
    if not quick_only and quick_total > 0:
        speedup = f"{naive_total / quick_total:.2f}x"

    print("\nPODSUMOWANIE")
    print(f"Liczba instancji: {len(instances)}")
    print(f"Błędne wyniki względem pliku porównawczego: {wrong}")
    print(f"Quick NEH total: {quick_total:.6f} s")
    print(f"Quick NEH max dla pojedynczej instancji: {quick_max:.6f} s")
    if not quick_only:
        print(f"NEH bez akceleracji total: {naive_total:.6f} s")
        print(f"NEH bez akceleracji max dla pojedynczej instancji: {naive_max:.6f} s")
        print(f"Przyspieszenie łączne Quick/Naive: {speedup}")
    print(f"CSV zapisano do: {csv_path}")


def validate_quick_speed(instances: List[Instance], threshold_s: float = 1.0) -> bool:
    """
    Waliduje warunek czasowy dla Quick NEH.

    Zwraca True, jeśli maksymalny czas dla wszystkich instancji jest mniejszy
    niż threshold_s. W przeciwnym razie zwraca False.
    """
    if not instances:
        print("FAIL: brak instancji do walidacji")
        return False

    if len(instances) != 120:
        print(f"UWAGA: walidacja uruchomiona na {len(instances)} instancjach, oczekiwano 120")

    # Rozgrzewka JIT nie wchodzi do pomiaru.
    neh_quick_numba(instances[0][1])

    quick_max = 0.0
    worst_instance = ""

    for instance_id, p, _, _ in instances:
        start = time.perf_counter()
        neh_quick_numba(p)
        quick_time = time.perf_counter() - start

        if quick_time > quick_max:
            quick_max = quick_time
            worst_instance = f"data.{instance_id:03d}"

    if quick_max < threshold_s:
        print(f"PASS: max quick NEH = {quick_max:.6f} s < {threshold_s:.3f} s ({worst_instance})")
        return True

    print(f"FAIL: max quick NEH = {quick_max:.6f} s >= {threshold_s:.3f} s ({worst_instance})")
    return False
