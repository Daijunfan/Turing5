from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from random import Random

from .dynamics import offline_optimum
from .pruning import local_context_candidate


@dataclass(frozen=True)
class TableTask:
    index: int


class TableModel:
    def __init__(
        self,
        service: tuple[tuple[int, ...], ...],
        transition: tuple[tuple[tuple[int, ...], ...], ...],
    ):
        self.service = service
        self.transition = transition

    @property
    def state_count(self) -> int:
        return len(self.service[0])

    def service_cost(self, task: TableTask, state: int) -> Fraction:
        return Fraction(self.service[task.index][state])

    def transition_cost(
        self, task: TableTask, previous: int | None, state: int
    ) -> Fraction:
        return Fraction(0 if previous is None else self.transition[task.index][previous][state])


@dataclass(frozen=True)
class ContextCounterexample:
    model: TableModel
    removed_state: int
    trace: tuple[TableTask, ...]
    full_cost: Fraction
    pruned_cost: Fraction


def find_local_context_counterexample(
    seed: int = 2,
    max_trials: int = 10_000,
    max_horizon: int = 6,
) -> ContextCounterexample:
    """Deterministically search the smallest trace length in a bounded model family."""

    random = Random(seed)
    tasks = (TableTask(0), TableTask(1))
    for _ in range(max_trials):
        service = tuple(tuple(random.randrange(9) for _ in range(3)) for _ in tasks)
        transition = tuple(
            tuple(
                tuple(0 if i == j else random.randrange(9) for j in range(3))
                for i in range(3)
            )
            for _ in tasks
        )
        model = TableModel(service, transition)
        if not local_context_candidate(model, 0, tasks):
            continue
        for length in range(1, max_horizon + 1):
            for trace in product(tasks, repeat=length):
                full = offline_optimum(model, trace)
                pruned = offline_optimum(model, trace, (1, 2))
                if (
                    full.cost is not None
                    and pruned.cost is not None
                    and pruned.cost > full.cost
                ):
                    return ContextCounterexample(
                        model, 0, tuple(trace), full.cost, pruned.cost
                    )
    raise RuntimeError("no context-rule counterexample found in the bounded search")

