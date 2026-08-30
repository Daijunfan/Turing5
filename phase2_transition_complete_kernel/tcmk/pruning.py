from __future__ import annotations

from fractions import Fraction
from typing import Hashable, Iterable, Sequence

from .dynamics import FiniteCostModel, add


def _leq(left: Fraction | None, right: Fraction | None) -> bool:
    if right is None:
        return True
    return left is not None and left <= right


def pareto_indices(features: Sequence[tuple[Fraction | None, ...]]) -> tuple[int, ...]:
    result = []
    for index, point in enumerate(features):
        dominated = False
        for other_index, other in enumerate(features):
            if other_index == index:
                continue
            weak = all(_leq(a, b) for a, b in zip(other, point))
            strict = any(a != b for a, b in zip(other, point))
            if weak and strict:
                dominated = True
                break
        if not dominated:
            result.append(index)
    return tuple(result)


def service_anchor_pareto(
    model: FiniteCostModel,
    task_alphabet: Sequence[Hashable],
    anchors: Sequence[int],
) -> tuple[int, ...]:
    """Pareto-prune service and directed migration distances to fixed anchors."""

    features: list[tuple[Fraction | None, ...]] = []
    for state in range(model.state_count):
        row: list[Fraction | None] = [
            model.service_cost(task, state) for task in task_alphabet
        ]
        for anchor in anchors:
            for task in task_alphabet:
                row.append(model.transition_cost(task, anchor, state))
                row.append(model.transition_cost(task, state, anchor))
        features.append(tuple(row))
    return pareto_indices(features)


def local_context_candidate(
    model: FiniteCostModel,
    state: int,
    task_alphabet: Sequence[Hashable],
) -> bool:
    """The proposed one-occurrence condition; this function does not claim safety."""

    alternatives = tuple(index for index in range(model.state_count) if index != state)
    for previous in range(model.state_count):
        for following in range(model.state_count):
            for task in task_alphabet:
                target = add(
                    add(
                        model.transition_cost(task, previous, state),
                        model.service_cost(task, state),
                    ),
                    model.transition_cost(task, state, following),
                )
                if not any(
                    _leq(
                        add(
                            add(
                                model.transition_cost(task, previous, other),
                                model.service_cost(task, other),
                            ),
                            model.transition_cost(task, other, following),
                        ),
                        target,
                    )
                    for other in alternatives
                ):
                    return False
    return True


def block_context_witness(
    model: FiniteCostModel,
    state: int,
    task_alphabet: Sequence[Hashable],
) -> int | None:
    """A sufficient deletion rule that safely replaces every maximal x-block.

    One fixed witness must dominate entry (including service), repeated stays,
    and exit transitions.  Using the same witness is the key condition missing
    from the one-occurrence proposal.
    """

    others = tuple(index for index in range(model.state_count) if index != state)
    for witness in others:
        entry_ok = all(
            _leq(
                add(
                    model.transition_cost(task, previous, witness),
                    model.service_cost(task, witness),
                ),
                add(
                    model.transition_cost(task, previous, state),
                    model.service_cost(task, state),
                ),
            )
            for previous in (None, *others)
            for task in task_alphabet
        )
        stay_ok = all(
            _leq(
                add(
                    model.transition_cost(task, witness, witness),
                    model.service_cost(task, witness),
                ),
                add(
                    model.transition_cost(task, state, state),
                    model.service_cost(task, state),
                ),
            )
            for task in task_alphabet
        )
        exit_ok = all(
            _leq(
                model.transition_cost(task, witness, following),
                model.transition_cost(task, state, following),
            )
            for following in others
            for task in task_alphabet
        )
        if entry_ok and stay_ok and exit_ok:
            return witness
    return None


def block_context_kernel(
    model: FiniteCostModel, task_alphabet: Sequence[Hashable]
) -> tuple[int, ...]:
    removable = {
        state
        for state in range(model.state_count)
        if block_context_witness(model, state, task_alphabet) is not None
    }
    # Simultaneous deletion can remove witnesses, so retain every used witness.
    witnesses = {
        block_context_witness(model, state, task_alphabet) for state in removable
    }
    removable -= {state for state in witnesses if state is not None}
    return tuple(state for state in range(model.state_count) if state not in removable)

