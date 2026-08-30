from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Sequence

from .morphology import Morphology


@dataclass(frozen=True)
class BudgetTask:
    budget: int


class SplitDistanceModel:
    """Legacy-only regression model; not evidence for the final TCMK claims."""

    def __init__(self, states: Sequence[Morphology], beta: int | Fraction):
        self.states = tuple(states)
        self.beta = Fraction(beta)
        self.base_work = min(state.work for state in states)
        internal_count = len(states[0].splits)
        self.unit = self.beta * self.base_work / max(1, 2 * internal_count)

    @property
    def state_count(self) -> int:
        return len(self.states)

    def service_cost(self, task: Hashable, state: int) -> Fraction | None:
        assert isinstance(task, BudgetTask)
        morphology = self.states[state]
        return Fraction(morphology.work) if morphology.peak <= task.budget else None

    def transition_cost(
        self, task: Hashable, previous: int | None, state: int
    ) -> Fraction | None:
        del task
        target = self.states[state].splits
        if previous is None:
            # Explicitly construct the first architecture from an empty source.
            return self.unit * len(target)
        return self.unit * len(self.states[previous].splits ^ target)

    def is_metric(self) -> bool:
        # Symmetric-difference cardinality is a metric; positive scaling preserves it.
        return self.unit >= 0

