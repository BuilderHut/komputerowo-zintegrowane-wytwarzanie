#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punkt startowy programu NEH / Quick NEH dla problemu F||Cmax.

Szczegóły implementacji są podzielone na moduły:
- data_io.py - wczytywanie instancji,
- neh_algorithms.py - algorytmy NEH i Quick NEH,
- benchmark.py - pomiary, CSV i walidacja czasu.
"""

from __future__ import annotations

import argparse
import sys

from benchmark import run_benchmark, validate_quick_speed
from data_io import parse_data_file


def main() -> None:
    parser = argparse.ArgumentParser(description="NEH i Quick NEH dla F||Cmax")
    parser.add_argument("data_file", nargs="?", default="data.000.txt", help="ścieżka do data.000.txt")
    parser.add_argument("--quick-only", action="store_true", help="uruchom tylko wersję z akceleracją")
    parser.add_argument(
        "--validate-speed",
        action="store_true",
        help="sprawdź, czy maksymalny czas Quick NEH dla 120 instancji jest poniżej 1 s",
    )
    parser.add_argument(
        "--include-data-000",
        action="store_true",
        help="uwzględnij małą instancję przykładową data.000; domyślnie liczone jest data.001-data.120",
    )
    parser.add_argument("--csv", default="wyniki_neh.csv", help="plik CSV z wynikami")
    args = parser.parse_args()

    try:
        instances = parse_data_file(args.data_file)
    except Exception as exc:
        print(f"Błąd wczytywania danych: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not args.include_data_000:
        instances = [item for item in instances if item[0] != 0]

    if not instances:
        print("Nie znaleziono żadnych instancji data.xxx", file=sys.stderr)
        raise SystemExit(1)

    if args.validate_speed:
        ok = validate_quick_speed(instances, threshold_s=1.0)
        raise SystemExit(0 if ok else 1)

    run_benchmark(instances, args.quick_only, args.csv)


if __name__ == "__main__":
    main()
