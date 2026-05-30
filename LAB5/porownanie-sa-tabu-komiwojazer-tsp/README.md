# Porownanie SA i Tabu Search dla TSP

Projekt porownuje Symulowane Wyzarzanie i Tabu Search dla problemu
komiwojazera. Funkcja celu to dlugosc zamknietej trasy przechodzacej przez
wszystkie miasta.

## Pliki

- `tsp_algorithms.py` - generowanie instancji, funkcja celu, SA i Tabu Search.
- `run_tsp_experiments.py` - glowne badania porownawcze.
- `run_iteration_experiment.py` - badanie wplywu liczby iteracji.
- `results/` - wyniki eksperymentow w CSV.
- `plots/` - wykresy PNG.
- `requirements.txt` - zaleznosci Pythona.

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie badan

```bash
python run_tsp_experiments.py
python run_iteration_experiment.py
```
