from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Hashable, Sequence

from .dynamics import Cost, FiniteCostModel
from .residual import SeparationResult, exact_separation


@dataclass(frozen=True)
class BridgeValue:
    state: int
    value: Cost | None
    witness: tuple[Hashable, ...]

    @property
    def infinite(self) -> bool:
        return self.value is None


def bridge_value(
    model: FiniteCostModel,
    state: int,
    task_alphabet: Sequence[Hashable],
    horizon: int,
    initial_state: int | None = None,
) -> BridgeValue:
    remaining = tuple(index for index in range(model.state_count) if index != state)
    result = exact_separation(model, task_alphabet, horizon, remaining, initial_state)
    return BridgeValue(state, result.gap, result.trace)


@dataclass(frozen=True)
class CEIteration:
    kernel_before: tuple[int, ...]
    separation: SeparationResult
    missing_path_states: tuple[int, ...]
    bridge_values: tuple[BridgeValue, ...]
    added_state: int | None


@dataclass(frozen=True)
class KernelCertificate:
    kernel: tuple[int, ...]
    horizon: int
    task_alphabet: tuple[Hashable, ...]
    initial_state: int | None
    final_separation: SeparationResult
    iterations: tuple[CEIteration, ...]
    deleted_states: tuple[int, ...]


def ce_tcmk(
    model: FiniteCostModel,
    task_alphabet: Sequence[Hashable],
    horizon: int,
    initial_kernel: Sequence[int],
    initial_state: int | None = None,
) -> KernelCertificate:
    kernel = set(initial_kernel)
    iterations: list[CEIteration] = []
    while True:
        separation = exact_separation(
            model, task_alphabet, horizon, tuple(kernel), initial_state
        )
        if separation.equivalent:
            break
        missing = tuple(dict.fromkeys(state for state in separation.full_path if state not in kernel))
        if not missing:
            raise RuntimeError("positive separation without a missing full-oracle path state")
        values = tuple(
            bridge_value(model, state, task_alphabet, horizon, initial_state)
            for state in missing
        )
        # Infinite bridge values are indispensable and rank above finite values.
        chosen = min(
            values,
            key=lambda item: (
                0 if item.infinite else 1,
                Fraction(0) if item.value is None else -item.value,
                item.state,
            ),
        ).state
        iterations.append(
            CEIteration(tuple(sorted(kernel)), separation, missing, values, chosen)
        )
        kernel.add(chosen)

    deleted: list[int] = []
    for state in tuple(sorted(kernel)):
        candidate = tuple(sorted(kernel - {state}))
        if not candidate:
            continue
        check = exact_separation(
            model,
            task_alphabet,
            horizon,
            candidate,
            initial_state,
            stop_on_positive=True,
        )
        if check.equivalent:
            kernel.remove(state)
            deleted.append(state)
    final = exact_separation(model, task_alphabet, horizon, tuple(kernel), initial_state)
    if not final.equivalent:
        raise AssertionError("CE-TCMK deletion pass broke dynamic equivalence")
    return KernelCertificate(
        tuple(sorted(kernel)),
        horizon,
        tuple(task_alphabet),
        initial_state,
        final,
        tuple(iterations),
        tuple(deleted),
    )


def minimum_kernel_oracle(
    model: FiniteCostModel,
    task_alphabet: Sequence[Hashable],
    horizon: int,
    initial_state: int | None = None,
) -> KernelCertificate:
    """Cardinality-minimum kernel by exhaustive subset enumeration."""

    universe = tuple(range(model.state_count))
    for size in range(1, len(universe) + 1):
        for candidate in combinations(universe, size):
            separation = exact_separation(
                model,
                task_alphabet,
                horizon,
                candidate,
                initial_state,
                stop_on_positive=True,
            )
            if separation.equivalent:
                separation = exact_separation(
                    model, task_alphabet, horizon, candidate, initial_state
                )
                return KernelCertificate(
                    candidate,
                    horizon,
                    tuple(task_alphabet),
                    initial_state,
                    separation,
                    (),
                    (),
                )
    raise AssertionError("the full state set must be transition-complete")
