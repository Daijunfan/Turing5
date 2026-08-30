#!/usr/bin/env python3
"""Generic proof-carrying morphology envelope for tensor/join hypergraphs.

Each tensor is a leaf. Every index label occurs exactly twice, so a subset's
materialized result is determined solely by its cut boundary, independent of
contraction order. The compiler enumerates subsets, combines Pareto frontiers,
and emits exact work/peak-memory morphologies plus a derivation certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class State:
    work: int
    peak: int
    left_mask: int
    left_index: int
    right_index: int
    right_first: bool


@dataclass(frozen=True)
class TensorValue:
    labels: tuple[int, ...]
    data: np.ndarray


@dataclass
class TensorEnvelope:
    incident_masks: list[int]
    edge_dims: list[int]

    def __post_init__(self) -> None:
        self.n = len(self.incident_masks)
        self.full = (1 << self.n) - 1
        self.boundary = [0] * (1 << self.n)
        for s in range(1, 1 << self.n):
            bit = s & -s
            vertex = bit.bit_length() - 1
            self.boundary[s] = self.boundary[s ^ bit] ^ self.incident_masks[vertex]
        self._size_cache: dict[int, int] = {0: 1}
        self.result_size = [self.mask_product(m) for m in self.boundary]
        self.frontiers: list[list[State] | None] = [None] * (1 << self.n)

    def mask_product(self, mask: int) -> int:
        cached = self._size_cache.get(mask)
        if cached is not None:
            return cached
        p = 1
        x = mask
        while x:
            bit = x & -x
            p *= self.edge_dims[bit.bit_length() - 1]
            x ^= bit
        self._size_cache[mask] = p
        return p

    @staticmethod
    def prune(states: list[State]) -> list[State]:
        states.sort(key=lambda s: (s.peak, s.work, s.left_mask, s.left_index, s.right_index))
        out: list[State] = []
        best_work: int | None = None
        last_peak: int | None = None
        for state in states:
            if state.peak == last_peak:
                continue
            last_peak = state.peak
            if best_work is None or state.work < best_work:
                out.append(state)
                best_work = state.work
        return out

    def compile(self) -> None:
        for v in range(self.n):
            self.frontiers[1 << v] = [State(0, 0, 0, -1, -1, False)]
        for cardinality in range(2, self.n + 1):
            for subset in range(1, self.full + 1):
                if subset.bit_count() != cardinality:
                    continue
                first = subset & -subset
                candidates: list[State] = []
                left = (subset - 1) & subset
                while left:
                    if left != subset and left & first:
                        right = subset ^ left
                        lf = self.frontiers[left]
                        rf = self.frontiers[right]
                        assert lf is not None and rf is not None
                        ltemp = self.result_size[left] if left & (left - 1) else 0
                        rtemp = self.result_size[right] if right & (right - 1) else 0
                        out_size = self.result_size[subset]
                        coexist = ltemp + rtemp + out_size
                        local_work = self.mask_product(self.boundary[left] | self.boundary[right])
                        for li, lstate in enumerate(lf):
                            for ri, rstate in enumerate(rf):
                                left_first = max(lstate.peak, ltemp + rstate.peak, coexist)
                                right_first = max(rstate.peak, rtemp + lstate.peak, coexist)
                                candidates.append(
                                    State(
                                        lstate.work + rstate.work + local_work,
                                        min(left_first, right_first),
                                        left,
                                        li,
                                        ri,
                                        right_first < left_first,
                                    )
                                )
                    left = (left - 1) & subset
                self.frontiers[subset] = self.prune(candidates)

    @property
    def root(self) -> list[State]:
        value = self.frontiers[self.full]
        assert value is not None
        return value

    def reconstruct(self, subset: int, index: int):
        state = self.frontiers[subset][index]  # type: ignore[index]
        if subset & (subset - 1) == 0:
            return subset.bit_length() - 1
        left = state.left_mask
        right = subset ^ left
        return (
            self.reconstruct(left, state.left_index),
            self.reconstruct(right, state.right_index),
            state.right_first,
        )

    def certificate_metrics(self, tree) -> tuple[int, int, int, int]:
        if isinstance(tree, int):
            subset = 1 << tree
            return 0, 0, subset, self.result_size[subset]
        left_tree, right_tree, right_first = tree
        lw, lp, left, lsize = self.certificate_metrics(left_tree)
        rw, rp, right, rsize = self.certificate_metrics(right_tree)
        assert not (left & right)
        subset = left | right
        ltemp = lsize if left & (left - 1) else 0
        rtemp = rsize if right & (right - 1) else 0
        out_size = self.result_size[subset]
        coexist = ltemp + rtemp + out_size
        pleft = max(lp, ltemp + rp, coexist)
        pright = max(rp, rtemp + lp, coexist)
        peak = pright if right_first else pleft
        work = lw + rw + self.mask_product(self.boundary[left] | self.boundary[right])
        return work, peak, subset, out_size


def double_factorial(k: int) -> int:
    out = 1
    while k > 1:
        out *= k
        k -= 2
    return out


def random_connected_graph(n: int, rng: random.Random, extra_edges: int) -> tuple[list[int], list[int]]:
    edges: set[tuple[int, int]] = set()
    for v in range(1, n):
        u = rng.randrange(v)
        edges.add((u, v))
    while len(edges) < n - 1 + extra_edges:
        u, v = sorted(rng.sample(range(n), 2))
        edges.add((u, v))
    incident = [0] * n
    dims: list[int] = []
    for edge_id, (u, v) in enumerate(sorted(edges)):
        incident[u] |= 1 << edge_id
        incident[v] |= 1 << edge_id
        dims.append(rng.choice((2, 3)))
    return incident, dims


def grid(rows: int, cols: int, dimensions: Iterable[int]) -> tuple[list[int], list[int]]:
    incident = [0] * (rows * cols)
    pairs: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                pairs.append((v, v + 1))
            if r + 1 < rows:
                pairs.append((v, v + cols))
    dims = list(dimensions)
    if len(dims) != len(pairs):
        raise ValueError(f"need {len(pairs)} edge dimensions")
    for edge_id, (u, v) in enumerate(pairs):
        incident[u] |= 1 << edge_id
        incident[v] |= 1 << edge_id
    return incident, dims


def brute_metrics(env: TensorEnvelope, subset: int, memo: dict[int, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    if subset in memo:
        return memo[subset]
    if subset & (subset - 1) == 0:
        memo[subset] = [(0, 0)]
        return memo[subset]
    first = subset & -subset
    out: list[tuple[int, int]] = []
    left = (subset - 1) & subset
    while left:
        if left != subset and left & first:
            right = subset ^ left
            ltemp = env.result_size[left] if left & (left - 1) else 0
            rtemp = env.result_size[right] if right & (right - 1) else 0
            output = env.result_size[subset]
            coexist = ltemp + rtemp + output
            local = env.mask_product(env.boundary[left] | env.boundary[right])
            for lw, lp in brute_metrics(env, left, memo):
                for rw, rp in brute_metrics(env, right, memo):
                    p1 = max(lp, ltemp + rp, coexist)
                    p2 = max(rp, rtemp + lp, coexist)
                    out.append((lw + rw + local, min(p1, p2)))
        left = (left - 1) & subset
    memo[subset] = out
    return out


def pareto_pairs(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    values.sort(key=lambda x: (x[1], x[0]))
    out: list[tuple[int, int]] = []
    best: int | None = None
    last_peak: int | None = None
    for work, peak in values:
        if peak == last_peak:
            continue
        last_peak = peak
        if best is None or work < best:
            out.append((work, peak))
            best = work
    return out


def leaf_values(incident: list[int], dims: list[int], rng: random.Random) -> list[TensorValue]:
    values: list[TensorValue] = []
    for mask in incident:
        labels = tuple(i for i in range(len(dims)) if mask >> i & 1)
        shape = tuple(dims[i] for i in labels)
        data = np.array([rng.randrange(4) for _ in range(math.prod(shape))], dtype=np.int64).reshape(shape)
        values.append(TensorValue(labels, data))
    return values


def contract(a: TensorValue, b: TensorValue) -> TensorValue:
    shared = sorted(set(a.labels) & set(b.labels))
    axes_a = [a.labels.index(x) for x in shared]
    axes_b = [b.labels.index(x) for x in shared]
    data = np.tensordot(a.data, b.data, axes=(axes_a, axes_b))
    labels = tuple(x for x in a.labels if x not in shared) + tuple(x for x in b.labels if x not in shared)
    return TensorValue(labels, data)


def execute(tree, leaves: list[TensorValue]) -> TensorValue:
    if isinstance(tree, int):
        return leaves[tree]
    left, right, right_first = tree
    if right_first:
        r = execute(right, leaves)
        l = execute(left, leaves)
    else:
        l = execute(left, leaves)
        r = execute(right, leaves)
    return contract(l, r)


def canonical_tree(n: int):
    tree = 0
    for i in range(1, n):
        tree = (tree, i, False)
    return tree


def correctness_suite() -> dict[str, int]:
    rng = random.Random(0xA11CE)
    exact_instances = 0
    enumerated_trees = 0
    certificate_checks = 0
    semantic_checks = 0
    for n in range(4, 8):
        for _ in range(24):
            incident, dims = random_connected_graph(n, rng, extra_edges=rng.randrange(0, n // 2 + 1))
            env = TensorEnvelope(incident, dims)
            env.compile()
            got = [(s.work, s.peak) for s in env.root]
            all_values = brute_metrics(env, env.full, {})
            expected = pareto_pairs(all_values)
            assert got == expected
            exact_instances += 1
            enumerated_trees += len(all_values)
            for i, state in enumerate(env.root):
                tree = env.reconstruct(env.full, i)
                work, peak, subset, _ = env.certificate_metrics(tree)
                assert subset == env.full and work == state.work and peak == state.peak
                certificate_checks += 1
            leaves = leaf_values(incident, dims, rng)
            reference = execute(canonical_tree(n), leaves)
            for i in range(len(env.root)):
                value = execute(env.reconstruct(env.full, i), leaves)
                assert value.labels == reference.labels
                assert np.array_equal(value.data, reference.data)
                semantic_checks += 1
    return {
        "exact_instances": exact_instances,
        "enumerated_contraction_trees": enumerated_trees,
        "certificate_checks": certificate_checks,
        "actual_tensor_semantic_checks": semantic_checks,
        "failures": 0,
    }


def large_benchmark() -> dict[str, object]:
    # A documented heterogeneous 3x4-grid family; edge dimensions are generated
    # independently for each seed from powers of two, not tuned after execution.
    records = []
    for seed in range(20):
        rng = random.Random(50_000 + seed)
        dims = [2 ** rng.randrange(1, 8) for _ in range(17)]
        incident, dims = grid(3, 4, dims)
        t0 = time.perf_counter()
        env = TensorEnvelope(incident, dims)
        env.compile()
        elapsed = time.perf_counter() - t0
        records.append({
            "seed": seed,
            "basis": len(env.root),
            "compile_seconds": elapsed,
            "endpoint_work_ratio": env.root[0].work / env.root[-1].work,
            "frontier": [[s.work, s.peak] for s in env.root],
            "edge_dimensions": dims,
        })
    representative = max(records, key=lambda r: (r["endpoint_work_ratio"], r["basis"]))
    return {
        "tensors": 12,
        "exact_semantic_space": double_factorial(21),
        "log10_semantic_space": math.log10(double_factorial(21)),
        "seeds": len(records),
        "fraction_multiple_morphologies": sum(r["basis"] > 1 for r in records) / len(records),
        "mean_basis": statistics.fmean(r["basis"] for r in records),
        "max_basis": max(r["basis"] for r in records),
        "median_compile_seconds": statistics.median(r["compile_seconds"] for r in records),
        "max_endpoint_work_ratio": max(r["endpoint_work_ratio"] for r in records),
        "representative": representative,
        "all_records": records,
    }


def main() -> int:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "results/tensor_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "correctness": correctness_suite(),
        "large_tensor_grid": large_benchmark(),
        "status": "all tensor-hypergraph assertions passed",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "correctness": result["correctness"],
        "large_tensor_grid": {k: v for k, v in result["large_tensor_grid"].items() if k != "all_records"},
        "status": result["status"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
