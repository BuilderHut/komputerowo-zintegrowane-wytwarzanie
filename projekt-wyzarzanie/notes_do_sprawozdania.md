# Notatki do sprawozdania

## Wstęp

Celem projektu jest zastosowanie symulowanego wyżarzania do permutacyjnego problemu przepływowego `F||Cmax` oraz porównanie wyników z algorytmem NEH.

## Problem F||Cmax

W problemie przepływowym każde zadanie przechodzi przez wszystkie maszyny w tej samej kolejności. Celem jest minimalizacja `Cmax`, czyli czasu zakończenia ostatniego zadania na ostatniej maszynie.

## NEH

NEH jest heurystyką konstrukcyjną. Sortuje zadania według sumy czasów operacji, a następnie iteracyjnie wstawia kolejne zadania w najlepsze miejsce częściowej permutacji. W projekcie NEH służy jako punkt odniesienia oraz jako dobre rozwiązanie startowe dla symulowanego wyżarzania.

## Symulowane wyżarzanie

Symulowane wyżarzanie startuje z pewnej permutacji, generuje sąsiada i decyduje, czy przejść do nowego rozwiązania. Lepsze rozwiązania są akceptowane zawsze, a gorsze z prawdopodobieństwem:

```text
exp(-(new_cmax - current_cmax) / T)
```

Dzięki temu algorytm może opuszczać minima lokalne. Gdy temperatura maleje, akceptacja gorszych rozwiązań staje się coraz mniej prawdopodobna.

## Parametry

Badano wpływ liczby iteracji, temperatury początkowej, współczynnika chłodzenia `alpha`, rozwiązania początkowego (`neh` lub `random`) oraz typu sąsiedztwa (`swap` lub `insert`).

## Wykresy

Wygenerowano wykres przebiegu `current_cmax` i `best_cmax`, wykresy wpływu parametrów oraz porównanie NEH i SA. Wykres przebiegu jest ważny, ponieważ pokazuje, że aktualne rozwiązanie może czasem się pogorszyć, ale najlepsze znalezione rozwiązanie pozostaje nierosnące.

## Wnioski

Symulowane wyżarzanie pozwala dalej przeszukiwać przestrzeń permutacji po uzyskaniu rozwiązania NEH. Jakość wyników zależy od parametrów algorytmu, szczególnie od liczby iteracji, tempa chłodzenia i sposobu generowania sąsiedztwa.
