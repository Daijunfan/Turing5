from __future__ import annotations

from typing import Sequence

import numpy as np

from .kernel import BridgeValue, CEIteration, KernelCertificate
from .residual import SeparationResult


INF = np.int64(1 << 60)


def offline_table(
    source: np.ndarray,
    transition: np.ndarray,
    trace: Sequence[int],
    allowed: Sequence[int],
) -> tuple[int | None, tuple[int, ...]]:
    if not trace:
        return 0, ()
    states = np.asarray(tuple(allowed), dtype=np.int64)
    costs = source[trace[0], states].copy()
    parents = []
    for task in trace[1:]:
        arrivals = costs[:, None] + transition[task][states][:, states]
        parent = np.argmin(arrivals, axis=0)
        costs = arrivals[parent, np.arange(len(states))]
        parents.append(parent)
    position = int(np.argmin(costs))
    if costs[position] >= INF:
        return None, ()
    path = [int(states[position])]
    for parent in reversed(parents):
        position = int(parent[position])
        path.append(int(states[position]))
    path.reverse()
    return int(min(costs)), tuple(path)


def _key(full: np.ndarray, kernel: np.ndarray, gap: int) -> tuple:
    return tuple(map(int, full)), tuple(map(int, kernel)), int(gap)


def fast_separation(
    source: np.ndarray,
    transition: np.ndarray,
    horizon: int,
    kernel: Sequence[int],
    *,
    stop_on_positive: bool = False,
) -> SeparationResult:
    """Integer/NumPy implementation of the same paired-residual search."""

    state_count = source.shape[1]
    full_ids = tuple(range(state_count))
    kernel_ids = tuple(sorted(set(kernel)))
    if not kernel_ids:
        return SeparationResult(None, (), (), (), (), True)
    current: dict[tuple, tuple[int, ...]] = {}
    best_gap = 0
    best_trace: tuple[int, ...] = ()
    for task in range(source.shape[0]):
        full = source[task].copy()
        full_min = int(np.min(full))
        if full_min >= INF:
            continue
        selected = source[task, kernel_ids].copy()
        kernel_min = int(np.min(selected))
        trace = (task,)
        if kernel_min >= INF:
            _, full_path = offline_table(source, transition, trace, full_ids)
            return SeparationResult(None, trace, full_path, (), (len(current),), True)
        full = np.where(full >= INF, INF, full - full_min)
        selected = np.where(selected >= INF, INF, selected - kernel_min)
        gap = kernel_min - full_min
        current.setdefault(_key(full, selected, gap), trace)
        if gap > best_gap:
            best_gap, best_trace = gap, trace
            if stop_on_positive:
                _, full_path = offline_table(source, transition, trace, full_ids)
                _, kernel_path = offline_table(source, transition, trace, kernel_ids)
                return SeparationResult(gap, trace, full_path, kernel_path, (len(current),))

    counts = [len(current)]
    kernel_array = np.asarray(kernel_ids, dtype=np.int64)
    for _depth in range(2, horizon + 1):
        keys = tuple(current)
        if not keys:
            counts.append(0)
            continue
        full_vectors = np.asarray([key[0] for key in keys], dtype=np.int64)
        kernel_vectors = np.asarray([key[1] for key in keys], dtype=np.int64)
        gaps = np.asarray([key[2] for key in keys], dtype=np.int64)
        prefixes = tuple(current[key] for key in keys)
        following: dict[tuple, tuple[int, ...]] = {}
        for task in range(source.shape[0]):
            full_raw = np.min(
                full_vectors[:, :, None] + transition[task][None, :, :], axis=1
            )
            full_min = np.min(full_raw, axis=1)
            kernel_matrix = transition[task][kernel_array][:, kernel_array]
            kernel_raw = np.min(
                kernel_vectors[:, :, None] + kernel_matrix[None, :, :], axis=1
            )
            kernel_min = np.min(kernel_raw, axis=1)
            full_norm = np.where(
                full_raw >= INF, INF, full_raw - full_min[:, None]
            )
            kernel_norm = np.where(
                kernel_raw >= INF, INF, kernel_raw - kernel_min[:, None]
            )
            new_gaps = gaps + kernel_min - full_min
            for row, prefix in enumerate(prefixes):
                trace = prefix + (task,)
                if full_min[row] >= INF:
                    continue
                if kernel_min[row] >= INF:
                    _, full_path = offline_table(source, transition, trace, full_ids)
                    return SeparationResult(
                        None,
                        trace,
                        full_path,
                        (),
                        tuple(counts + [len(following)]),
                        True,
                    )
                gap = int(new_gaps[row])
                residual = _key(full_norm[row], kernel_norm[row], gap)
                following.setdefault(residual, trace)
                if gap > best_gap:
                    best_gap, best_trace = gap, trace
                    if stop_on_positive:
                        _, full_path = offline_table(source, transition, trace, full_ids)
                        _, kernel_path = offline_table(
                            source, transition, trace, kernel_ids
                        )
                        return SeparationResult(
                            gap,
                            trace,
                            full_path,
                            kernel_path,
                            tuple(counts + [len(following)]),
                        )
        current = following
        counts.append(len(current))

    if not best_trace:
        best_trace = next(iter(current.values()), ())
    _, full_path = offline_table(source, transition, best_trace, full_ids)
    _, kernel_path = offline_table(source, transition, best_trace, kernel_ids)
    return SeparationResult(
        best_gap, best_trace, full_path, kernel_path, tuple(counts)
    )


def fast_bridge_values(
    source: np.ndarray,
    transition: np.ndarray,
    horizon: int,
    states: Sequence[int],
) -> tuple[BridgeValue, ...]:
    universe = tuple(range(source.shape[1]))
    values = []
    for state in states:
        result = fast_separation(
            source,
            transition,
            horizon,
            tuple(item for item in universe if item != state),
        )
        values.append(BridgeValue(state, result.gap, result.trace))
    return tuple(values)


def fast_ce_tcmk(
    source: np.ndarray,
    transition: np.ndarray,
    horizon: int,
    initial_kernel: Sequence[int],
) -> KernelCertificate:
    kernel = set(initial_kernel)
    iterations = []
    while True:
        separation = fast_separation(source, transition, horizon, tuple(kernel))
        if separation.equivalent:
            break
        missing = tuple(
            dict.fromkeys(state for state in separation.full_path if state not in kernel)
        )
        if not missing:
            raise RuntimeError("positive separation without a missing state")
        values = fast_bridge_values(source, transition, horizon, missing)
        chosen = min(
            values,
            key=lambda item: (
                0 if item.value is None else 1,
                0 if item.value is None else -item.value,
                item.state,
            ),
        ).state
        iterations.append(
            CEIteration(tuple(sorted(kernel)), separation, missing, values, chosen)
        )
        kernel.add(chosen)
    deleted = []
    for state in tuple(sorted(kernel)):
        candidate = tuple(sorted(kernel - {state}))
        if candidate and fast_separation(
            source, transition, horizon, candidate, stop_on_positive=True
        ).equivalent:
            kernel.remove(state)
            deleted.append(state)
    final = fast_separation(source, transition, horizon, tuple(kernel))
    return KernelCertificate(
        tuple(sorted(kernel)),
        horizon,
        tuple(range(source.shape[0])),
        None,
        final,
        tuple(iterations),
        tuple(deleted),
    )
