import re
import sys
import time

from .metoda_carliera import carlier
from .metody_schrage import popraw_kolejnosc_przez_wstawianie, schrage


def wczytaj_wszystkie_instancje(plik):
    with open(plik, "r", encoding="utf-8") as uchwyt:
        wiersze = uchwyt.readlines()

    instancje = []
    wzorzec = re.compile(r"^\s*data[.:]\d+\s*$")
    i = 0

    while i < len(wiersze):
        nazwa = wiersze[i].strip()
        if not wzorzec.fullmatch(nazwa):
            i += 1
            continue

        i += 1
        while i < len(wiersze) and not wiersze[i].strip():
            i += 1
        if i >= len(wiersze):
            break

        liczba_zadan = int(wiersze[i].strip())
        i += 1
        zadania = []

        for numer in range(1, liczba_zadan + 1):
            r, p, q = map(int, wiersze[i].split())
            zadania.append({"nr": numer, "r": r, "p": p, "q": q})
            i += 1

        instancje.append((nazwa, zadania))

    return instancje


def uruchom():
    plik = "rpq.data.txt"
    tryb = "dokladnie"
    inst = None
    limit = 60.0

    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--schrage" or a == "--schrage-only":
            tryb = "schrage"
        elif a == "--heur" or a == "--heurystyka":
            tryb = "heurystyka"
        elif a == "--fast" or a == "--szybko":
            tryb = "heurystyka"
        elif a == "--exact" or a == "--dokladnie":
            tryb = "dokladnie"
        elif (a == "--time-limit" or a == "--limit-czasu") and i + 1 < len(sys.argv):
            limit = float(sys.argv[i + 1])
            i += 1
        elif a == "--no-time-limit" or a == "--bez-limitu":
            limit = None
        elif (a == "--instance" or a == "--instancja") and i + 1 < len(sys.argv):
            inst = sys.argv[i + 1]
            i += 1
        elif not a.startswith("--"):
            plik = a
        else:
            print(f"Nieznany argument: {a}", file=sys.stderr)
            return 1
        i += 1

    try:
        instancje = wczytaj_wszystkie_instancje(plik)
    except OSError:
        print(f"Nie udalo sie otworzyc pliku: {plik}", file=sys.stderr)
        return 1

    if not instancje:
        print(f"Nie udalo sie wczytac instancji z pliku: {plik}", file=sys.stderr)
        return 1

    if inst is not None:
        tmp = []
        for jedna in instancje:
            if jedna[0] == inst:
                tmp.append(jedna)
        instancje = tmp
        if len(instancje) == 0:
            print(f"Nie znaleziono instancji: {inst}", file=sys.stderr)
            return 1

    suma = 0

    for nazwa, zadania in instancje:
        t0 = time.perf_counter()
        best_pi = []
        best = 0
        stop = False

        if tryb == "schrage":
            best_pi, best = schrage(zadania)
        elif tryb == "heurystyka":
            best_pi, best = popraw_kolejnosc_przez_wstawianie(zadania)
        else:
            pi0, ub = popraw_kolejnosc_przez_wstawianie(zadania)
            best = ub
            best_pi = pi0[:]
            best, best_pi, stop = carlier(
                zadania,
                pi0,
                ub,
                limit,
            )

        czas = time.perf_counter() - t0
        suma += best

        print(nazwa)
        print("Cmax =", best)
        if tryb == "dokladnie" and stop:
            print("status: limit czasu osiagniety, zwrocono najlepsze znalezione rozwiazanie")
        print(f"czas = {czas:.3f} s")
        print("kolejnosc:", end=" ")
        for indeks in best_pi:
            print(zadania[indeks]["nr"], end=" ")
        print()
        print()

    print("SUMA =", suma)
    return 0
