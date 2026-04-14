def solve_witi(z):
    n = len(z)
    m = 1 << len(z)
    P, F = [0] * m, [0] + [10**18] * (m - 1)
    for s in range(1, m):
        b = s & -s
        P[s] = P[s ^ b] + z[b.bit_length() - 1][0]

    calkowity_czas = P[m - 1]
    for s in range(1, m):
        # Zbior s traktujemy jako pozostaly sufiks harmonogramu.
        czas_startu_sufiksu = calkowity_czas - P[s]
        t = s
        while t:
            b = t & -t
            j = b.bit_length() - 1
            zakonczenie = czas_startu_sufiksu + z[j][0]
            v = z[j][1] * max(zakonczenie - z[j][2], 0) + F[s ^ b]
            if v < F[s]:
                F[s] = v
            t ^= b

    pi, s = [], m - 1
    while s:
        czas_startu_sufiksu = calkowity_czas - P[s]
        for j in range(n):
            b = 1 << j
            if not s & b:
                continue
            zakonczenie = czas_startu_sufiksu + z[j][0]
            v = z[j][1] * max(zakonczenie - z[j][2], 0) + F[s ^ b]
            if v == F[s]:
                pi.append(j + 1)
                s ^= b
                break
    return F[-1], pi


# Tworzymy 2 tablice:
# P[s] = laczny czas zadan ze zbioru s
# F[s] = najlepsza kara dla zbioru s jako sufiksu harmonogramu
# Kolejnosc odtwarzamy bez tablicy pamietajacej poprzednikow.
