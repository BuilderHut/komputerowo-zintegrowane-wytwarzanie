# RPQ - Schrage i Carlier

Program rozwiazuje problem szeregowania `1|r_j|Cmax` dla danych z pliku `rpq.data.txt`.

W projekcie sa wykorzystane:
- algorytm Schrage
- Schrage z przerwaniami
- algorytm Carliera
- prosta poprawa rozwiazania przez wstawianie

Program pozwala porownac 3 warianty:
- sam Schrage
- Schrage + dodatkowa heurystyka poprawiajaca
- Carlier

## Pliki

- `uruchom_rpq.py` - glowny plik do uruchamiania programu
- `rpq_solver/uruchomienie.py` - wczytywanie danych, argumentow i wypisywanie wynikow
- `rpq_solver/metody_schrage.py` - Schrage, Schrage z przerwaniami, poprawa przez wstawianie
- `rpq_solver/metoda_carliera.py` - algorytm Carliera
- `rpq.data.txt` - plik z instancjami

## Uruchamianie

Podstawowe uruchomienie:

```powershell
python uruchom_rpq.py
```

Z podaniem pliku:

```powershell
python uruchom_rpq.py rpq.data.txt
```

Sam Schrage:

```powershell
python uruchom_rpq.py rpq.data.txt --schrage
```

Heurystyka poprawiona:

```powershell
python uruchom_rpq.py rpq.data.txt --fast
```

To samo mozna uruchomic tez jako:

```powershell
python uruchom_rpq.py rpq.data.txt --heur
```

Tryb dokladny z Carlierem:

```powershell
python uruchom_rpq.py rpq.data.txt --exact
```

Jedna instancja:

```powershell
python uruchom_rpq.py rpq.data.txt --instance data.1
```

## Limit czasu

Domyslnie w trybie dokladnym z Carlierem limit czasu wynosi `60` sekund.

Wlasny limit:

```powershell
python uruchom_rpq.py rpq.data.txt --exact --time-limit 30
```

Bez limitu:

```powershell
python uruchom_rpq.py rpq.data.txt --exact --no-time-limit
```

## Wynik

Program wypisuje dla kazdej instancji:
- nazwe instancji
- wartosc `Cmax`
- czas obliczen
- kolejnosc zadan

Na koncu wypisywana jest tez suma wynikow dla wszystkich instancji.

## Krotka interpretacja trybow

- `--schrage` - uruchamia tylko podstawowy algorytm Schrage
- `--fast` lub `--heur` - uruchamia Schrage i dodatkowa heurystyke poprawiajaca
- `--exact` - uruchamia wariant heurystyczny, a potem Carliera
