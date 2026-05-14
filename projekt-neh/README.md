# NEH, Quick NEH i symulowane wyżarzanie dla F||Cmax

Projekt dotyczy permutacyjnego problemu przepływowego `F||Cmax`. Zaimplementowano:

- klasyczny algorytm NEH,
- przyspieszony Quick NEH / qNEH,
- symulowane wyżarzanie dla tego samego problemu.

Programy korzystają z tego samego pliku danych `data.000.txt`, zawierającego instancje `data.000` - `data.120`.

## Zawartość

- `neh_python_50.py` - punkt startowy dla NEH i Quick NEH
- `data_io.py` - wczytywanie instancji i wyników referencyjnych
- `neh_algorithms.py` - obliczanie `Cmax`, NEH i Quick NEH
- `benchmark.py` - benchmark NEH, zapis CSV i walidacja czasu
- `simulated_annealing.py` - pojedyncze uruchomienie symulowanego wyżarzania
- `run_sa_experiments.py` - badania porównawcze SA i generowanie wykresów
- `data.000.txt` - dane wejściowe
- `wyniki_neh.csv` - wyniki porównania NEH i Quick NEH
- `results/` - wyniki eksperymentów SA w CSV
- `plots/` - wykresy wygenerowane przez eksperymenty SA
- `notes_do_sprawozdania.md` - notatki do przygotowania sprawozdania
- `requirements.txt` - wymagane biblioteki

## Wymagania

- Python 3.10+
- `numpy`
- `numba`
- `pandas`
- `matplotlib`

Instalacja:

```bash
pip install -r requirements.txt
```

## Uruchomienie NEH

Domyślne uruchomienie dla instancji `data.001` - `data.120`:

```bash
python neh_python_50.py data.000.txt
```

Tylko Quick NEH:

```bash
python neh_python_50.py data.000.txt --quick-only
```

Walidacja warunku czasu poniżej 1 s dla Quick NEH:

```bash
python neh_python_50.py data.000.txt --validate-speed
```

## Uruchomienie symulowanego wyżarzania

Przykład dla jednej instancji:

```bash
python simulated_annealing.py data.000.txt --instance data.001 --iterations 10000 --t-start 1000 --t-end 0.01 --alpha 0.995 --initial neh --neighborhood swap --seed 123
```

Najważniejsze parametry:

- `--instance` - wybrana instancja, np. `data.001`
- `--iterations` - liczba iteracji
- `--t-start` - temperatura początkowa
- `--t-end` - temperatura końcowa
- `--alpha` - współczynnik chłodzenia
- `--initial` - rozwiązanie startowe: `neh` albo `random`
- `--neighborhood` - typ sąsiedztwa: `swap` albo `insert`
- `--seed` - ziarno losowości dla powtarzalności wyników

Historia iteracji jest zapisywana domyślnie do:

```text
results/sa_history_example.csv
```

## Badania eksperymentalne

Uruchomienie badań dla zakresu instancji:

```bash
python run_sa_experiments.py data.000.txt --instances data.001:data.020 --repeats 5
```

Skrypt bada wpływ:

- liczby iteracji,
- temperatury początkowej,
- współczynnika chłodzenia `alpha`,
- rozwiązania początkowego,
- typu sąsiedztwa.

Wyniki CSV:

- `results/sa_runs.csv` - wszystkie pojedyncze uruchomienia
- `results/sa_summary.csv` - uśrednione wyniki dla wariantów
- `results/comparison_neh_sa.csv` - porównanie NEH i SA
- `results/sa_history_example.csv` - historia przykładowego przebiegu SA

Wykresy:

- `plots/cmax_current_vs_best_example.png`
- `plots/influence_iterations.png`
- `plots/influence_temperature.png`
- `plots/influence_alpha.png`
- `plots/influence_initial.png`
- `plots/influence_neighborhood.png`
- `plots/comparison_neh_sa.png`

## Aktualnie wygenerowane wyniki

W repozytorium znajdują się przykładowe wyniki dla instancji `data.001` - `data.005`
z dwoma powtórzeniami na wariant. Pełne badania można uruchomić poleceniem z sekcji
`Badania eksperymentalne`.

## Uwagi

- Domyślnie NEH analizuje instancje `data.001` - `data.120`, czyli pomija małą instancję przykładową `data.000`.
- Symulowane wyżarzanie może startować z rozwiązania NEH albo z losowej permutacji.
- SA czasem akceptuje gorsze rozwiązania, co jest widoczne na wykresie `current_cmax` względem `best_cmax`.
- Projekt nie wymaga internetu podczas działania po zainstalowaniu zależności.
