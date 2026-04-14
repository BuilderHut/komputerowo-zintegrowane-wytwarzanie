import sys
import time

from witi_io import wczytaj_wszystkie_instancje
from witi_solver import solve_witi


def main():
    plik = sys.argv[1] if len(sys.argv) > 1 else "witi.data.txt"
    instancja = sys.argv[2].rstrip(":") if len(sys.argv) > 2 else None

    suma = 0
    for nazwa, zadania, opt, _ in wczytaj_wszystkie_instancje(plik):
        if instancja and nazwa != instancja:
            continue
        t = time.perf_counter()
        wynik, kolejnosc = solve_witi(zadania)
        suma += wynik
        print(nazwa)
        print("K =", wynik)
        if opt is not None:
            print("opt =", opt)
            print("zgodnosc =", "tak" if wynik == opt else "nie")
        print(f"czas = {time.perf_counter() - t:.3f} s")
        print("kolejnosc:", *kolejnosc)
        print()

    if not instancja:
        print("SUMA =", suma)


if __name__ == "__main__":
    main()
