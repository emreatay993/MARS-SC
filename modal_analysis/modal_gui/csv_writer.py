"""CSV writer helpers for modal extraction."""

from __future__ import annotations

import csv
from typing import Iterable, Sequence, TextIO

import numpy as np


def write_header(handle: TextIO, header: Sequence[str]) -> None:
    handle.write(",".join(header) + "\n")


def write_chunk(handle: TextIO, data: np.ndarray) -> None:
    """Write a 2D numpy array chunk to CSV with stable float formatting."""
    if data.size == 0:
        return
    n_cols = data.shape[1]
    fmt = ["%d"] + ["%.15g"] * (n_cols - 1)
    np.savetxt(handle, data, delimiter=",", fmt=fmt)


def iter_csv_rows(path: str) -> Iterable[list[str]]:
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            yield row
