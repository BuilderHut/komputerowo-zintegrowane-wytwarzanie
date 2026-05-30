from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


Instance = Tuple[int, np.ndarray, int, np.ndarray]


def parse_data_file(path: str | Path, require_reference: bool = True) -> List[Instance]:
    """Wczytuje wszystkie instancje data.xxx z jednego pliku data.000.txt."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Plik zawiera wiele bloków data.xxx. Wyrażenie regularne rozcina tekst tak,
    # żeby numer instancji i jej treść można było przetwarzać parami.
    parts = re.split(r"(?m)^data\.(\d+):\s*$", text)

    instances: List[Instance] = []
    for i in range(1, len(parts), 2):
        instance_id = int(parts[i])
        lines = [line.strip() for line in parts[i + 1].splitlines() if line.strip()]
        if not lines:
            continue

        n, m = map(int, lines[0].split())
        processing_times = np.array(
            [list(map(int, lines[1 + r].split())) for r in range(n)],
            dtype=np.int64,
        )

        reference_cmax = -1
        reference_sequence: List[int] = []
        for k in range(1 + n, len(lines)):
            if lines[k].lower().startswith("neh"):
                reference_cmax = int(lines[k + 1].split()[0])
                values: List[int] = []
                for line in lines[k + 2 :]:
                    values.extend(map(int, line.split()))
                reference_sequence = [x - 1 for x in values[:n]]  # 0-based
                break

        if require_reference and (reference_cmax < 0 or len(reference_sequence) != n):
            raise ValueError(f"Brak poprawnego wyniku referencyjnego NEH dla data.{instance_id:03d}")

        instances.append(
            (
                instance_id,
                processing_times,
                reference_cmax,
                np.array(reference_sequence, dtype=np.int64),
            )
        )

    return instances
