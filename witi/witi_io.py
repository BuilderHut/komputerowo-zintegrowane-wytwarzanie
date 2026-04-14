def wczytaj_wszystkie_instancje(plik):
    w = [x.strip() for x in open(plik, encoding="utf-8") if x.strip()]
    instancje, i = [], 0
    while i < len(w):
        if not w[i].startswith(("data.", "data:")):
            i += 1
            continue
        n = int(w[i + 1])
        j = i + 2 + n
        opt = int(w[j + 1]) if j < len(w) and w[j] == "opt:" else None
        pi = list(map(int, w[j + 2].split())) if opt is not None else None
        instancje.append((w[i].rstrip(":"), [tuple(map(int, x.split())) for x in w[i + 2 : j]], opt, pi))
        i = j + 3 if opt is not None else j
    return instancje
