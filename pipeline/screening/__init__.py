import logging
import numpy as np
from typing import TypeVar, Generic
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Filter(ABC, Generic[T]):
    name: str = "base_filter"
    input_label: str = "input"
    output_label: str = "output"

    @abstractmethod
    def score(self, candidate: T) -> float:
        pass

    @abstractmethod
    def passes(self, candidate: T, threshold: float) -> bool:
        pass

    def filter_batch(self, candidates: list[T], threshold: float) -> list[T]:
        scored = []
        for c in candidates:
            score = self.score(c)
            if self.passes(c, threshold):
                scored.append(c)
        return scored

    def filter_stream(self, candidates_iter, threshold: float, target_count: int):
        total_tested = 0
        total_passed = 0
        for batch in candidates_iter:
            passed = self.filter_batch(batch, threshold)
            total_tested += len(batch)
            total_passed += len(passed)
            logger.info(
                "%s: %d/%d passed (%.2f%%) | Total: %d/%d",
                self.name, len(passed), len(batch),
                100 * len(passed) / max(len(batch), 1),
                total_passed, total_tested,
            )
            yield passed
            if total_passed >= target_count:
                break
