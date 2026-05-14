from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit
except ImportError as exc:
    raise SystemExit(
        "Brakuje biblioteki numba. Zainstaluj: pip install -r requirements.txt\n"
        "Bez numby Pythonowa wersja może nie spełnić wymogu czasu dla akceleracji."
    ) from exc


@njit(cache=True)
def sort_jobs_by_weight(p: np.ndarray) -> np.ndarray:
    """Sortowanie zadań nierosnąco po sumie czasów; przy remisie mniejszy numer zadania."""
    n, m = p.shape
    order = np.empty(n, dtype=np.int64)
    weights = np.empty(n, dtype=np.int64)

    for i in range(n):
        order[i] = i
        total = 0
        for k in range(m):
            total += p[i, k]
        weights[i] = total

    # NEH zaczyna od zadań o największym łącznym czasie przetwarzania.
    # Przy remisie wybieram mniejszy numer zadania, żeby wynik był deterministyczny
    # i zgodny z kolejnością referencyjną w danych.
    # n <= 500, więc prosty insertion sort jest wystarczający i dobrze wspierany przez numbę.
    for i in range(1, n):
        key = order[i]
        key_weight = weights[key]
        j = i - 1
        while j >= 0 and (
            weights[order[j]] < key_weight
            or (weights[order[j]] == key_weight and order[j] > key)
        ):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    return order


@njit(cache=True)
def cmax_numba(p: np.ndarray, sequence: np.ndarray, length: int) -> int:
    """Oblicza Cmax dla podanej kolejności zadań."""
    _, m = p.shape
    completion = np.zeros(m, dtype=np.int64)

    for i in range(length):
        job = sequence[i]

        # Na pierwszej maszynie zadania wykonują się jedno po drugim, więc czas
        # zakończenia jest tylko sumą kolejnych czasów operacji.
        completion[0] += p[job, 0]

        for machine in range(1, m):
            # Operacja może wystartować dopiero wtedy, gdy:
            # 1. to samo zadanie skończyło się na poprzedniej maszynie,
            # 2. ta maszyna skończyła poprzednie zadanie.
            # Dlatego bierzemy maksimum z obu czasów i dodajemy czas operacji.
            if completion[machine] < completion[machine - 1]:
                completion[machine] = completion[machine - 1] + p[job, machine]
            else:
                completion[machine] = completion[machine] + p[job, machine]

    return int(completion[m - 1])


@njit(cache=True)
def neh_naive_numba(p: np.ndarray) -> Tuple[int, np.ndarray]:
    """Klasyczny NEH bez akceleracji: dla każdej pozycji liczy pełne Cmax od nowa."""
    n, _ = p.shape
    order = sort_jobs_by_weight(p)

    sequence = np.empty(n, dtype=np.int64)
    candidate = np.empty(n, dtype=np.int64)
    length = 0

    for order_index in range(n):
        job = order[order_index]
        best_value = np.int64(9223372036854775807)
        best_position = 0

        # Sprawdzamy wszystkie możliwe miejsca wstawienia nowego zadania
        # do już zbudowanej częściowej permutacji.
        for position in range(length + 1):
            for i in range(position):
                candidate[i] = sequence[i]
            candidate[position] = job
            for i in range(position, length):
                candidate[i + 1] = sequence[i]

            # W wersji klasycznej dla każdej kandydackiej permutacji liczymy
            # całe Cmax od początku, co jest proste, ale kosztowne.
            value = cmax_numba(p, candidate, length + 1)
            if value < best_value:
                best_value = value
                best_position = position

        # Po wybraniu najlepszej pozycji aktualizujemy właściwą permutację.
        for i in range(length, best_position, -1):
            sequence[i] = sequence[i - 1]
        sequence[best_position] = job
        length += 1

    return cmax_numba(p, sequence, n), sequence.copy()


@njit(cache=True)
def neh_quick_numba(p: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    NEH z akceleracją qNEH.

    Dla aktualnej permutacji liczymy tablice:
    - e: najdłuższe ścieżki dochodzące,
    - q: najdłuższe ścieżki wychodzące.

    Dzięki temu koszt każdego próbnego wstawienia liczymy w O(m), a nie w O(n*m).
    """
    n, m = p.shape
    order = sort_jobs_by_weight(p)

    sequence = np.empty(n, dtype=np.int64)
    e = np.empty((n, m), dtype=np.int64)
    q = np.empty((n, m), dtype=np.int64)
    length = 0

    for order_index in range(n):
        job = order[order_index]

        if length == 0:
            sequence[0] = job
            length = 1
            continue

        # e[i, k] - najdłuższa ścieżka dochodząca do operacji (i, k), z czasem tej operacji
        for i in range(length):
            current_job = sequence[i]
            for machine in range(m):
                previous_job_value = 0
                if i > 0:
                    previous_job_value = e[i - 1, machine]

                previous_machine_value = 0
                if machine > 0:
                    previous_machine_value = e[i, machine - 1]

                if previous_machine_value > previous_job_value:
                    previous_job_value = previous_machine_value

                # e przechowuje czas zakończenia operacji dla obecnej częściowej
                # permutacji, liczony od lewej strony harmonogramu.
                e[i, machine] = previous_job_value + p[current_job, machine]

        # q[i, k] - najdłuższa ścieżka wychodząca z operacji (i, k), z czasem tej operacji
        for i in range(length - 1, -1, -1):
            current_job = sequence[i]
            for machine in range(m - 1, -1, -1):
                next_job_value = 0
                if i < length - 1:
                    next_job_value = q[i + 1, machine]

                next_machine_value = 0
                if machine < m - 1:
                    next_machine_value = q[i, machine + 1]

                if next_machine_value > next_job_value:
                    next_job_value = next_machine_value

                # q jest odpowiednikiem e liczonym od końca permutacji. Dzięki temu
                # znamy koszt "prawej strony" harmonogramu po planowanym wstawieniu.
                q[i, machine] = next_job_value + p[current_job, machine]

        best_value = np.int64(9223372036854775807)
        best_position = 0

        # Sprawdzamy wszystkie możliwe pozycje wstawienia zadania job.
        for position in range(length + 1):
            previous_inserted_machine = 0
            candidate_cmax = 0

            for machine in range(m):
                left_value = 0
                if position > 0:
                    left_value = e[position - 1, machine]

                # inserted_value opisuje czas zakończenia wstawianego zadania
                # na kolejnych maszynach. Musi uwzględniać zarówno lewą część
                # harmonogramu, jak i poprzednią maszynę tego samego zadania.
                if previous_inserted_machine > left_value:
                    inserted_value = previous_inserted_machine + p[job, machine]
                else:
                    inserted_value = left_value + p[job, machine]

                previous_inserted_machine = inserted_value

                right_value = 0
                if position < length:
                    right_value = q[position, machine]

                # Dla danej maszyny pełna długość ścieżki przechodzi przez:
                # lewą część, wstawiane zadanie oraz prawą część. Maksimum po
                # maszynach jest wartością Cmax dla tej pozycji wstawienia.
                value = inserted_value + right_value
                if value > candidate_cmax:
                    candidate_cmax = value

            if candidate_cmax < best_value:
                best_value = candidate_cmax
                best_position = position

        for i in range(length, best_position, -1):
            sequence[i] = sequence[i - 1]
        sequence[best_position] = job
        length += 1

    return cmax_numba(p, sequence, n), sequence.copy()
