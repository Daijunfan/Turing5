from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Sequence

from .dynamics import (
    Cost,
    FiniteCostModel,
    advance_vector,
    initial_vector,
    minimum,
    offline_optimum,
)


Vector = tuple[Cost | None, ...]


@dataclass(frozen=True)
class ResidualState:
    full: Vector
    kernel: Vector
    gap: Cost


@dataclass(frozen=True)
class SeparationResult:
    gap: Cost | None
    trace: tuple[Hashable, ...]
    full_path: tuple[int, ...]
    kernel_path: tuple[int, ...]
    reachable_by_depth: tuple[int, ...]
    kernel_infeasible: bool = False

    @property
    def equivalent(self) -> bool:
        return not self.kernel_infeasible and self.gap == 0


def _normalize(vector: Vector) -> tuple[Vector, Cost | None]:
    base = minimum(vector)
    if base is None:
        return vector, None
    return tuple(None if value is None else value - base for value in vector), base


def _initial_residual(
    model: FiniteCostModel,
    task: Hashable,
    full: tuple[int, ...],
    kernel: tuple[int, ...],
    initial_state: int | None,
) -> ResidualState | None:
    full_raw = initial_vector(model, task, full, initial_state)
    kernel_raw = initial_vector(model, task, kernel, initial_state)
    full_norm, full_min = _normalize(full_raw)
    kernel_norm, kernel_min = _normalize(kernel_raw)
    if full_min is None:
        return None
    if kernel_min is None:
        return ResidualState(full_norm, kernel_norm, Fraction(0))
    return ResidualState(full_norm, kernel_norm, kernel_min - full_min)


def _advance_residual(
    model: FiniteCostModel,
    residual: ResidualState,
    task: Hashable,
    full: tuple[int, ...],
    kernel: tuple[int, ...],
) -> ResidualState | None:
    full_raw = advance_vector(model, task, full, residual.full, full)
    kernel_raw = advance_vector(model, task, kernel, residual.kernel, kernel)
    full_norm, full_shift = _normalize(full_raw)
    kernel_norm, kernel_shift = _normalize(kernel_raw)
    if full_shift is None:
        return None
    if kernel_shift is None:
        return ResidualState(full_norm, kernel_norm, residual.gap)
    return ResidualState(
        full_norm,
        kernel_norm,
        residual.gap + kernel_shift - full_shift,
    )


def exact_separation(
    model: FiniteCostModel,
    task_alphabet: Sequence[Hashable],
    horizon: int,
    kernel: Sequence[int],
    initial_state: int | None = None,
    *,
    stop_on_positive: bool = False,
) -> SeparationResult:
    """Find the exact maximum ``OPT_K - OPT_H`` up to ``horizon``.

    Search states are paired min-plus residual vectors.  Each vector has its
    minimum subtracted, and identical paired residuals (including their offset
    gap) are merged.  This enumerates reachable residual states, not task words.
    """

    if horizon < 1:
        return SeparationResult(Fraction(0), (), (), (), ())
    full = tuple(range(model.state_count))
    kernel = tuple(sorted(set(kernel)))
    if not kernel:
        return SeparationResult(None, (), (), (), (), True)

    current: dict[ResidualState, tuple[Hashable, ...]] = {}
    best_gap = Fraction(0)
    best_trace: tuple[Hashable, ...] = ()
    for task in task_alphabet:
        residual = _initial_residual(model, task, full, kernel, initial_state)
        if residual is None:
            continue
        trace = (task,)
        if minimum(residual.kernel) is None:
            full_result = offline_optimum(model, trace, full, initial_state)
            return SeparationResult(None, trace, full_result.path, (), (len(current),), True)
        current.setdefault(residual, trace)
        if residual.gap > best_gap:
            best_gap, best_trace = residual.gap, trace
            if stop_on_positive:
                full_result = offline_optimum(model, trace, full, initial_state)
                kernel_result = offline_optimum(model, trace, kernel, initial_state)
                return SeparationResult(
                    best_gap,
                    trace,
                    full_result.path,
                    kernel_result.path,
                    (len(current),),
                )

    counts = [len(current)]
    for _depth in range(2, horizon + 1):
        following: dict[ResidualState, tuple[Hashable, ...]] = {}
        for residual, prefix in current.items():
            for task in task_alphabet:
                advanced = _advance_residual(model, residual, task, full, kernel)
                if advanced is None:
                    continue
                trace = prefix + (task,)
                if minimum(advanced.kernel) is None:
                    full_result = offline_optimum(model, trace, full, initial_state)
                    return SeparationResult(
                        None,
                        trace,
                        full_result.path,
                        (),
                        tuple(counts + [len(following)]),
                        True,
                    )
                following.setdefault(advanced, trace)
                if advanced.gap > best_gap:
                    best_gap, best_trace = advanced.gap, trace
                    if stop_on_positive:
                        full_result = offline_optimum(model, trace, full, initial_state)
                        kernel_result = offline_optimum(
                            model, trace, kernel, initial_state
                        )
                        return SeparationResult(
                            best_gap,
                            trace,
                            full_result.path,
                            kernel_result.path,
                            tuple(counts + [len(following)]),
                        )
        current = following
        counts.append(len(current))

    if not best_trace:
        # Any feasible representative is a valid zero-gap certificate witness.
        best_trace = next(iter(current.values()), ())
    full_result = offline_optimum(model, best_trace, full, initial_state)
    kernel_result = offline_optimum(model, best_trace, kernel, initial_state)
    return SeparationResult(
        best_gap,
        best_trace,
        full_result.path,
        kernel_result.path,
        tuple(counts),
    )
