# Symulowane wyżarzanie dla F||Cmax

Projekt dotyczy permutacyjnego problemu przepływowego `F||Cmax`. Rozwiązania są porównywane z algorytmem NEH / Quick NEH, a główną częścią projektu jest algorytm symulowanego wyżarzania.

## Zawartość

- `simulated_annealing.py` - pojedyncze uruchomienie symulowanego wyżarzania
- `run_sa_experiments.py` - badania porównawcze, CSV i wykresy
- `neh_algorithms.py` - funkcja `Cmax`, NEH i Quick NEH używane jako punkt odniesienia
- `data_io.py` - wczytywanie instancji `data.xxx`
- `data.000.txt` - dane wejściowe
- `results/` - wyniki badań w CSV
- `plots/` - wykresy PNG
- `notes_do_sprawozdania.md` - notatki do sprawozdania
- `requirements.txt` - zależności

## Instalacja

```bash
pip install -r requirements.txt
```

## Pojedyncze uruchomienie SA

```bash
python simulated_annealing.py data.000.txt --instance data.001 --iterations 10000 --t-start 1000 --t-end 0.01 --alpha 0.995 --initial neh --neighborhood swap --seed 123
```

Parametry:

- `--instance` - instancja, np. `data.001`
- `--iterations` - liczba iteracji
- `--t-start` - temperatura początkowa
- `--t-end` - temperatura końcowa
- `--alpha` - współczynnik chłodzenia
- `--initial` - `neh` albo `random`
- `--neighborhood` - `swap` albo `insert`
- `--seed` - ziarno losowości

## Eksperymenty

```bash
python run_sa_experiments.py data.000.txt --instances data.001:data.020 --repeats 5
```

Skrypt bada wpływ:

- liczby iteracji,
- temperatury początkowej,
- współczynnika `alpha`,
- rozwiązania początkowego,
- typu sąsiedztwa.

## Wyniki

Pliki CSV:

- `results/sa_runs.csv`
- `results/sa_summary.csv`
- `results/comparison_neh_sa.csv`
- `results/sa_history_example.csv`

Wykresy:

- `plots/cmax_current_vs_best_example.png`
- `plots/influence_iterations.png`
- `plots/influence_temperature.png`
- `plots/influence_alpha.png`
- `plots/influence_initial.png`
- `plots/influence_neighborhood.png`
- `plots/comparison_neh_sa.png`

W repozytorium znajdują się przykładowe wyniki dla zakresu `data.001:data.005` i `repeats=2`. Pełne badania można uruchomić komendą z sekcji `Eksperymenty`.
