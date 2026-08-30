from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Sequence

from .matrix_chain import Tree, result_size


NodeBias = Callable[[int, int, int], int]


@dataclass(frozen=True)
class PriceState:
    memory_elements: int
    additive_cost: int
    tree: Tree


@dataclass(frozen=True)
class PricingResult:
    tree: Tree
    total_cost: int
    target_memory_elements: int
    source_memory_elements: int
    peak_memory_bytes: int
    dp_states_generated: int
    dp_states_retained: int
    exact: bool
    certificate_digest: str


class PricingResourceLimit(RuntimeError):
    def __init__(self, generated: int, retained: int, limit: int):
        super().__init__(f"pricing DP state limit {limit} exceeded")
        self.generated = generated
        self.retained = retained
        self.limit = limit


def _pareto(states: list[PriceState]) -> tuple[PriceState, ...]:
    best_by_memory: dict[int, PriceState] = {}
    for state in states:
        previous = best_by_memory.get(state.memory_elements)
        if previous is None or state.additive_cost < previous.additive_cost:
            best_by_memory[state.memory_elements] = state
    result = []
    best_cost = None
    for memory in sorted(best_by_memory):
        state = best_by_memory[memory]
        if best_cost is None or state.additive_cost < best_cost:
            result.append(state)
            best_cost = state.additive_cost
    return tuple(result)


def price_transition(
    dims: Sequence[int],
    source: Tree,
    update: int,
    memory_budget_bytes: int | None = None,
    node_bias: NodeBias | None = None,
    max_generated_states: int | None = None,
) -> PricingResult:
    """Exact interval DP for the best target tree from one fixed predecessor."""

    n = len(dims) - 1
    source_intervals = source.intervals()
    source_memory = sum(result_size(dims, *interval) for interval in source_intervals)
    cells: dict[tuple[int, int], tuple[PriceState, ...]] = {}
    generated = retained = 0
    for index in range(n):
        cells[index, index] = (PriceState(0, 0, Tree(index, index)),)
        retained += 1
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            candidates = []
            output = result_size(dims, i, j)
            reusable = (i, j) in source_intervals and not (i <= update <= j)
            for split in range(i, j):
                local = 0
                if not reusable:
                    work = dims[i] * dims[split + 1] * dims[j + 1]
                    reads = result_size(dims, i, split) + result_size(dims, split + 1, j)
                    writes = output
                    local = work + reads + writes
                if node_bias is not None:
                    local += node_bias(i, split, j)
                pair_count = len(cells[i, split]) * len(cells[split + 1, j])
                if (
                    max_generated_states is not None
                    and generated + len(candidates) + pair_count
                    > max_generated_states
                ):
                    raise PricingResourceLimit(
                        generated + len(candidates), retained, max_generated_states
                    )
                for left in cells[i, split]:
                    for right in cells[split + 1, j]:
                        tree = Tree(i, j, left.tree, right.tree)
                        candidates.append(
                            PriceState(
                                left.memory_elements + right.memory_elements + output,
                                left.additive_cost + right.additive_cost + local,
                                tree,
                            )
                        )
            generated += len(candidates)
            cells[i, j] = _pareto(candidates)
            retained += len(cells[i, j])
    feasible = tuple(
        state
        for state in cells[0, n - 1]
        if memory_budget_bytes is None
        or max(source_memory, state.memory_elements) * 8 <= memory_budget_bytes
    )
    if not feasible:
        raise ValueError("no target architecture satisfies the migration memory budget")
    chosen = min(
        feasible,
        key=lambda state: (
            state.additive_cost + max(source_memory, state.memory_elements),
            state.memory_elements,
            state.tree.render(),
        ),
    )
    total = chosen.additive_cost + max(source_memory, chosen.memory_elements)
    certificate = {
        "dims": tuple(dims),
        "source": source.render(),
        "update": update,
        "budget": memory_budget_bytes,
        "root_frontier": tuple(
            (state.memory_elements, state.additive_cost, state.tree.render())
            for state in cells[0, n - 1]
        ),
        "chosen": (chosen.memory_elements, chosen.additive_cost, chosen.tree.render()),
        "total": total,
    }
    return PricingResult(
        chosen.tree,
        total,
        chosen.memory_elements,
        source_memory,
        max(source_memory, chosen.memory_elements) * 8,
        generated,
        retained,
        True,
        sha256(
            json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
