# WITI

Ten folder zawiera komplet plikow dla zadania WITI:

- `witi_solver.py` - algorytm dynamiczny po podzbiorach
- `witi_io.py` - wczytywanie instancji z pliku
- `witi_pd.py` - uruchamianie programu i wypisywanie wynikow
- `witi.data.txt` - dane testowe

Solver:

- zwraca minimalna sume WITI,
- zwraca pierwsza leksykograficznie optymalna permutacje,
- nie uzywa dodatkowej tablicy zapamietujacej kolejnosc albo poprzednikow.

Uruchomienie:

```powershell
python witi_pd.py
python witi_pd.py witi.data.txt data.10
```
