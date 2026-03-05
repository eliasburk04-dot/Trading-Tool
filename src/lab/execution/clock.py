from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import pandas as pd


@dataclass(frozen=True)
class ReplayClock:
    timestamps: Iterable[pd.Timestamp]

    def __iter__(self) -> Iterator[pd.Timestamp]:
        return iter(self.timestamps)


@dataclass(frozen=True)
class LivePaperClock:
    timestamps: Iterable[pd.Timestamp]
    max_iterations: int = 1

    def __iter__(self) -> Iterator[pd.Timestamp]:
        for index, timestamp in enumerate(self.timestamps):
            if index >= self.max_iterations:
                break
            yield timestamp
