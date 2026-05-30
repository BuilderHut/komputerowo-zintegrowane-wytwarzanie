# Porownanie SA i Tabu Search dla F||Cmax

Projekt porownuje dwie metaheurystyki dla permutacyjnego problemu przeplywowego
`F||Cmax`. Celem jest minimalizacja czasu zakonczenia ostatniego zadania na
ostatniej maszynie.

## Pliki

- `simulated_annealing.py` - implementacja Symulowanego Wyzarzania.
- `tabu_search.py` - implementacja Tabu Search.
- `run_comparison_experiments.py` - glowne badania porownawcze.
- `run_iteration_experiment.py` - badanie wplywu liczby iteracji.
- `data_io.py`, `neh_algorithms.py` - wczytywanie danych, liczenie `Cmax`, NEH.
- `data.000.txt` - zestaw instancji.
- `results/` - wyniki eksperymentow w CSV.
- `plots/` - wykresy PNG.
- `requirements.txt` - zaleznosci Pythona.

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie badan

```bash
python run_comparison_experiments.py data.000.txt --instances data.111:data.120 --repeats 3
python run_iteration_experiment.py
```

## Pojedyncze uruchomienia

```bash
python simulated_annealing.py data.000.txt --instance data.120 --iterations 10000 --t-start 1000 --alpha 0.995 --initial random
python tabu_search.py data.000.txt --instance data.120 --iterations 500 --tabu-tenure 15 --candidates 40 --initial random
```
