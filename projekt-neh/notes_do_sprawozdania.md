# Notatki do sprawozdania

## 1. Wstęp

Celem projektu jest rozwiązanie permutacyjnego problemu przepływowego `F||Cmax` z użyciem algorytmu NEH, jego wersji przyspieszonej oraz algorytmu symulowanego wyżarzania. Dodatkowo wykonano badania porównawcze, które pozwalają sprawdzić wpływ parametrów symulowanego wyżarzania na jakość uzyskanych harmonogramów.

## 2. Opis problemu F||Cmax

W problemie przepływowym każde zadanie musi zostać wykonane na wszystkich maszynach w tej samej kolejności technologicznej. Szukana jest taka permutacja zadań, która minimalizuje `Cmax`, czyli moment zakończenia ostatniego zadania na ostatniej maszynie. Dla ustalonej permutacji czas zakończenia operacji można liczyć rekurencyjnie jako maksimum z czasu zakończenia poprzedniego zadania na tej samej maszynie oraz czasu zakończenia tego samego zadania na poprzedniej maszynie.

## 3. Opis algorytmu NEH

Algorytm NEH jest heurystyką konstrukcyjną. Najpierw sortuje zadania nierosnąco według sumy czasów przetwarzania na wszystkich maszynach. Następnie buduje permutację krok po kroku, wstawiając kolejne zadanie we wszystkie możliwe pozycje i wybierając tę, która daje najmniejszy `Cmax`. W projekcie zaimplementowano także Quick NEH, który używa tablic pomocniczych `e` i `q`, aby szybciej oceniać koszt wstawienia zadania.

## 4. Opis symulowanego wyżarzania

Symulowane wyżarzanie jest metaheurystyką inspirowaną procesem chłodzenia materiału. Algorytm startuje z pewnej permutacji zadań, na przykład z wyniku NEH albo z permutacji losowej. W każdej iteracji generuje sąsiednie rozwiązanie przez zamianę dwóch zadań miejscami albo przez wyjęcie zadania i wstawienie go w inne miejsce.

Jeżeli sąsiad jest lepszy, zostaje zaakceptowany. Jeżeli jest gorszy, może zostać zaakceptowany z prawdopodobieństwem:

```text
exp(-(new_cmax - current_cmax) / T)
```

Akceptacja gorszych rozwiązań pozwala algorytmowi wychodzić z minimów lokalnych. Na początku, gdy temperatura jest wysoka, takie ruchy są bardziej prawdopodobne. Wraz ze spadkiem temperatury algorytm coraz rzadziej akceptuje pogorszenia i zachowuje się bardziej zachłannie.

## 5. Temperatura i chłodzenie

Temperatura `T` steruje prawdopodobieństwem przyjmowania gorszych rozwiązań. W projekcie zastosowano chłodzenie geometryczne:

```text
T = T * alpha
```

Parametr `alpha` powinien być mniejszy od 1. Im bliżej 1 znajduje się `alpha`, tym wolniejsze chłodzenie i dłuższe przeszukiwanie z możliwością akceptacji pogorszeń.

## 6. Dokumentacja programu

Projekt jest podzielony na moduły:

- `data_io.py` - wczytywanie danych,
- `neh_algorithms.py` - NEH, Quick NEH i funkcja `Cmax`,
- `simulated_annealing.py` - pojedyncze uruchomienie SA,
- `run_sa_experiments.py` - eksperymenty, CSV i wykresy,
- `benchmark.py` - benchmark NEH.

Program zapisuje wyniki eksperymentów do katalogu `results/`, a wykresy do katalogu `plots/`.

## 7. Badania porównawcze

W badaniach analizowano wpływ następujących parametrów:

- liczba iteracji,
- temperatura początkowa,
- współczynnik chłodzenia `alpha`,
- rozwiązanie początkowe: NEH albo losowe,
- typ sąsiedztwa: `swap` albo `insert`.

Dla każdego wariantu wykonywano powtórzenia z różnymi ziarnami losowości. Wyniki porównano z wartością `Cmax` uzyskaną przez NEH.

## 8. Wykresy

Wygenerowano wykres przebiegu `current_cmax` oraz `best_cmax` po iteracjach. Jest on istotny, ponieważ pokazuje, że aktualne rozwiązanie może czasem się pogorszyć, gdy algorytm zaakceptuje gorszego sąsiada, natomiast najlepsze znalezione rozwiązanie pozostaje nierosnące.

Dodatkowo wygenerowano wykresy wpływu liczby iteracji, temperatury początkowej, parametru `alpha`, rozwiązania początkowego i typu sąsiedztwa, a także porównanie wartości `Cmax` uzyskanych przez NEH i symulowane wyżarzanie.

## 9. Wnioski

NEH jest szybkim algorytmem konstrukcyjnym i daje dobre rozwiązanie startowe. Symulowane wyżarzanie może wykorzystać wynik NEH jako punkt początkowy i dalej przeszukiwać przestrzeń permutacji. Jakość wyniku SA zależy od parametrów, szczególnie od liczby iteracji, temperatury początkowej, tempa chłodzenia oraz typu sąsiedztwa.

## 10. Literatura

W sprawozdaniu można odwołać się do materiałów dotyczących:

- problemów przepływowych,
- heurystyki NEH,
- metaheurystyki symulowanego wyżarzania,
- harmonogramowania produkcji.
