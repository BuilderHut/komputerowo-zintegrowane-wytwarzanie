import heapq
import sys

def policz_cmax(tab, kolej):
    t = 0
    cmax = 0
    for i in kolej:
        z = tab[i]
        if t < z["r"]:
            t = z["r"]
        t += z["p"]
        if t + z["q"] > cmax:
            cmax = t + z["q"]
    return cmax


def schrage(tab):
    niegotowe = []
    for i in range(len(tab)):
        z = tab[i]
        niegotowe.append((z["r"], i, z["p"], z["q"]))
    heapq.heapify(niegotowe)

    gotowe = []
    czas = niegotowe[0][0] if niegotowe else 0
    cmax = 0
    pi = []

    while gotowe or niegotowe:
        while niegotowe and niegotowe[0][0] <= czas:
            r, i, p, q = heapq.heappop(niegotowe)
            heapq.heappush(gotowe, (-q, i, p, r))

        if not gotowe:
            czas = niegotowe[0][0]
            continue

        minus_q, i, p, _ = heapq.heappop(gotowe)
        pi.append(i)
        czas += p
        q = -minus_q
        if czas + q > cmax:
            cmax = czas + q

    return pi, cmax


def schrage_z_przerwaniami(tab):
    niegotowe = []
    for i in range(len(tab)):
        z = tab[i]
        niegotowe.append((z["r"], i, z["p"], z["q"]))
    heapq.heapify(niegotowe)

    gotowe = []
    czas = niegotowe[0][0] if niegotowe else 0
    cmax = 0
    akt_id = -1
    akt_r = 0
    akt_q = sys.maxsize

    while gotowe or niegotowe:
        while niegotowe and niegotowe[0][0] <= czas:
            r, i, p, q = heapq.heappop(niegotowe)
            heapq.heappush(gotowe, (-q, i, r, p))

            if q > akt_q:
                zost = czas - r
                czas = r
                if zost > 0:
                    heapq.heappush(gotowe, (-akt_q, akt_id, akt_r, zost))

        if not gotowe:
            czas = niegotowe[0][0]
            continue

        minus_q, i, r, p = heapq.heappop(gotowe)
        akt_id = i
        akt_r = r
        akt_q = -minus_q
        czas += p
        if czas + akt_q > cmax:
            cmax = czas + akt_q

    return cmax


def popraw_kolejnosc_przez_wstawianie(tab):
    pi, ub = schrage(tab)

    poprawa = True
    while poprawa:
        poprawa = False

        for i in range(len(pi)):
            x = pi[i]
            reszta = pi[:i] + pi[i + 1 :]
            best = ub
            best_pi = pi

            for j in range(len(reszta) + 1):
                tmp = reszta[:]
                tmp.insert(j, x)
                val = policz_cmax(tab, tmp)
                if val < best:
                    best = val
                    best_pi = tmp

            if best < ub:
                ub = best
                pi = best_pi
                poprawa = True

    return pi, ub
