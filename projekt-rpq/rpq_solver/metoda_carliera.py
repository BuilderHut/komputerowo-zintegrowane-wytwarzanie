import sys
import time

from .metody_schrage import schrage, schrage_z_przerwaniami


class LimitCzasu(Exception):
    pass


def znajdz_blok_krytyczny(tab, pi, cmax):
    zak = [0] * len(pi)
    czas = 0

    for i in range(len(pi)):
        z = tab[pi[i]]
        if czas < z["r"]:
            czas = z["r"]
        czas += z["p"]
        zak[i] = czas

    b = -1
    for i in range(len(pi)):
        if zak[i] + tab[pi[i]]["q"] == cmax:
            b = i
    if b == -1:
        return -1, -1, -1

    a = -1
    suma_p = 0
    qq = tab[pi[b]]["q"]
    for i in range(b, -1, -1):
        suma_p += tab[pi[i]]["p"]
        if tab[pi[i]]["r"] + suma_p + qq == cmax:
            a = i

    c = -1
    for i in range(a, b):
        if tab[pi[i]]["q"] < qq:
            c = i

    return a, b, c


def policz_granice(tab, idx, r1, q1, p1):
    x = tab[idx]
    # trzy standardowe ograniczenia dolne
    h = schrage_z_przerwaniami(tab)
    h1 = r1 + p1 + q1
    h2 = min(x["r"], r1) + x["p"] + p1 + min(x["q"], q1)
    if h1 > h:
        h = h1
    if h2 > h:
        h = h2
    return h


def carlier(tab, start_pi, start_ub, limit_czasu=60.0):
    UB = start_ub
    best_pi = start_pi[:]
    odw = set()
    stop = None
    if limit_czasu is not None:
        stop = time.perf_counter() + max(0.0, limit_czasu)

    def licz():
        nonlocal UB, best_pi

        if stop is not None and time.perf_counter() >= stop:
            raise LimitCzasu

        stan = []
        for z in tab:
            stan.append((z["r"], z["q"]))
        stan = tuple(stan)
        if stan in odw:
            return
        odw.add(stan)

        pi, U = schrage(tab)
        if U < UB:
            UB = U
            best_pi = pi[:]

        LB = schrage_z_przerwaniami(tab)
        if LB >= UB:
            return

        a, b, c = znajdz_blok_krytyczny(tab, pi, U)
        if a == -1 or b == -1 or c == -1:
            return

        r1 = sys.maxsize
        q1 = sys.maxsize
        p1 = 0
        for i in range(c + 1, b + 1):
            z = tab[pi[i]]
            if z["r"] < r1:
                r1 = z["r"]
            if z["q"] < q1:
                q1 = z["q"]
            p1 += z["p"]

        idx = pi[c]
        stary_r = tab[idx]["r"]
        stary_q = tab[idx]["q"]

        tab[idx]["r"] = max(tab[idx]["r"], r1 + p1)
        nowy_r = tab[idx]["r"]
        lb1 = policz_granice(tab, idx, r1, q1, p1)
        tab[idx]["r"] = stary_r

        tab[idx]["q"] = max(tab[idx]["q"], q1 + p1)
        nowy_q = tab[idx]["q"]
        lb2 = policz_granice(tab, idx, r1, q1, p1)
        tab[idx]["q"] = stary_q

        opcje = [(lb1, 0, nowy_r), (lb2, 1, nowy_q)]
        opcje.sort()

        for lb, typ, nowa in opcje:
            if lb >= UB:
                continue

            if typ == 0:
                if nowa == stary_r:
                    continue
                tab[idx]["r"] = nowa
                licz()
                tab[idx]["r"] = stary_r
            else:
                if nowa == stary_q:
                    continue
                tab[idx]["q"] = nowa
                licz()
                tab[idx]["q"] = stary_q

    koniec = False
    try:
        licz()
    except LimitCzasu:
        koniec = True

    return UB, best_pi, koniec
