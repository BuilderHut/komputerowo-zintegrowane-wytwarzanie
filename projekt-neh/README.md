# NEH / Quick NEH dla F||Cmax

Projekt zawiera implementację algorytmu NEH dla problemu harmonogramowania przepływowego `F||Cmax`:

- klasyczny NEH bez akceleracji,
- przyspieszony Quick NEH / qNEH.

Skrypt porównuje wyniki z wartościami referencyjnymi zapisanymi w pliku `data.000.txt` i zapisuje zestawienie do CSV.

## Zawartość

- `neh_python_50.py` - główny plik uruchomieniowy
- `data_io.py` - wczytywanie instancji i wyników referencyjnych
- `neh_algorithms.py` - implementacja NEH oraz Quick NEH
- `benchmark.py` - pomiary czasu, zapis CSV i walidacja
- `data.000.txt` - plik z instancjami testowymi `data.000` - `data.120`
- `wyniki_neh.csv` - wyniki badań porównawczych dla 120 instancji
- `sprawozdanie.tex` - sprawozdanie w formacie LaTeX
- `requirements.txt` - lista wymaganych bibliotek Pythona

## Wymagania

- Python 3.10+
- `numpy`
- `numba`

Instalacja zależności:

```bash
pip install -r requirements.txt
```

## Uruchomienie

Domyślne uruchomienie:

```bash
python neh_python_50.py data.000.txt
```

Tylko wersja przyspieszona:

```bash
python neh_python_50.py data.000.txt --quick-only
```

Walidacja warunku czasowego dla Quick NEH:

```bash
python neh_python_50.py data.000.txt --validate-speed
```

Tryb ten wypisuje `PASS`, jeśli maksymalny czas Quick NEH dla 120 instancji jest mniejszy niż 1 s.

Uwzględnienie również instancji `data.000`:

```bash
python neh_python_50.py data.000.txt --include-data-000
```

Zmiana nazwy pliku CSV:

```bash
python neh_python_50.py data.000.txt --csv wyniki_neh.csv
```

## Opis działania

Program:

1. Wczytuje wszystkie instancje `data.xxx` z pliku wejściowego.
2. Pobiera z sekcji `neh:` wynik referencyjny.
3. Uruchamia:
   - `neh_quick_numba()` - wariant z akceleracją,
   - `neh_naive_numba()` - wariant klasyczny.
4. Porównuje `Cmax` oraz kolejność zadań z wynikiem referencyjnym.
5. Zapisuje wyniki do pliku CSV.

Kod jest podzielony na moduły, aby oddzielić logikę algorytmu od wczytywania danych
i części pomiarowej. Plik `neh_python_50.py` pozostaje punktem startowym programu.

## Wyniki

W konsoli pojawia się tabela z polami:

- `instance`
- `n`
- `m`
- `ref`
- `quick`
- `quick_time_s`
- `naive`
- `naive_time_s`
- `ok`

Plik CSV zawiera dodatkowo:

- sekwencję zadań dla Quick NEH,
- sekwencję zadań dla NEH bez akceleracji,
- flagę zgodności z wynikiem referencyjnym.

## Uwagi

- Skrypt zakłada, że w pliku wejściowym znajdują się sekcje w formacie `data.xxx:` oraz odpowiadające im wyniki `neh:`.
- Wersja Quick NEH jest przygotowana do porównań czasowych na większym zbiorze instancji.
- Bez `numba` skrypt nie uruchomi się, ponieważ obie implementacje korzystają z JIT.
- Folder `.vscode` nie jest wymagany do działania projektu; zawiera tylko lokalne ustawienia edytora.
