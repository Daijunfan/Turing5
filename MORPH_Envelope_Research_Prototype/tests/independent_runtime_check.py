#!/usr/bin/env python3
"""Independent Python validation of the C++ MORPH-Envelope prototype.

This file deliberately does not import or call the C++ implementation. It checks:
1. the exact two-point envelope for [1,H,H^3,H];
2. the local-rotation counterexample;
3. versioned drain-and-cutover execution with 10,000 real modular matrix jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Condition, Lock, Thread
from pathlib import Path
import json
import random
import sys
from typing import TypeAlias

MOD = 1_000_000_007
Tree: TypeAlias = int | tuple["Tree", "Tree"]
Matrix: TypeAlias = list[list[int]]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    assert a and b and len(a[0]) == len(b)
    out = [[0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            for j in range(len(b[0])):
                out[i][j] = (out[i][j] + aik * b[k][j]) % MOD
    return out


def eval_tree(t: Tree, xs: list[Matrix]) -> Matrix:
    if isinstance(t, int):
        return xs[t]
    return matmul(eval_tree(t[0], xs), eval_tree(t[1], xs))


def left_fold(xs: list[Matrix]) -> Matrix:
    z = xs[0]
    for x in xs[1:]:
        z = matmul(z, x)
    return z


def span(t: Tree) -> tuple[int, int]:
    if isinstance(t, int):
        return t, t
    return span(t[0])[0], span(t[1])[1]


def work(t: Tree, d: tuple[int, ...]) -> int:
    if isinstance(t, int):
        return 0
    l, r = t
    i, k = span(l)
    _, j = span(r)
    return work(l, d) + work(r, d) + d[i] * d[k + 1] * d[j + 1]


def rotations(t: Tree) -> set[Tree]:
    if isinstance(t, int):
        return set()
    l, r = t
    out: set[Tree] = set()
    if not isinstance(l, int):
        a, b = l
        out.add((a, (b, r)))
    if not isinstance(r, int):
        b, c = r
        out.add(((l, b), c))
    for x in rotations(l):
        out.add((x, r))
    for x in rotations(r):
        out.add((l, x))
    return out


def all_trees(i: int, j: int) -> list[Tree]:
    if i == j:
        return [i]
    out: list[Tree] = []
    for k in range(i, j):
        for l in all_trees(i, k):
            for r in all_trees(k + 1, j):
                out.append((l, r))
    return out


def verify_local_trap() -> dict[str, int | str]:
    d = (1, 1, 1, 2, 2)
    trap: Tree = (0, ((1, 2), 3))
    optimum = min(work(t, d) for t in all_trees(0, 3))
    trap_cost = work(trap, d)
    neighbor_min = min(work(t, d) for t in rotations(trap))
    assert optimum == 7 and trap_cost == 8 and neighbor_min == 8
    return {"global_optimum": optimum, "local_trap": trap_cost, "best_one_rotation": neighbor_min}


def verify_separation() -> list[dict[str, int | float]]:
    rows = []
    for h in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        low_work = h**5 + h**2
        high_work = 2 * h**4
        low_peak = h**2 + h
        high_peak = h**3 + h
        # One low-budget epoch followed by H high-budget epochs; two cutovers,
        # each charged H^3 state units.
        fixed = (h + 1) * low_work
        morph = low_work + h * high_work + 2 * h**3
        rows.append({
            "H": h,
            "work_ratio": low_work / high_work,
            "peak_ratio": high_peak / low_peak,
            "dynamic_static_over_morph": fixed / morph,
        })
    assert all(rows[i]["dynamic_static_over_morph"] < rows[i + 1]["dynamic_static_over_morph"] for i in range(len(rows) - 1))
    return rows


@dataclass(frozen=True)
class Task:
    seq: int
    version: int
    plan: Tree
    matrices: list[Matrix]


class VersionedRuntime:
    def __init__(self, plan: Tree, workers: int = 4):
        self.plan = plan
        self.version = 0
        self.accepting = True
        self.pending = 0
        self.cv = Condition()
        self.queue: Queue[Task | None] = Queue()
        self.results: dict[int, tuple[int, Matrix]] = {}
        self.result_lock = Lock()
        self.errors: list[str] = []
        self.threads = [Thread(target=self._worker, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def submit(self, seq: int, matrices: list[Matrix]) -> None:
        with self.cv:
            while not self.accepting:
                self.cv.wait()
            task = Task(seq, self.version, self.plan, matrices)
            self.pending += 1
        self.queue.put(task)

    def reconfigure(self, new_plan: Tree) -> None:
        # Proof-carrying cutover protocol: close ingress, drain all old-version
        # jobs, atomically replace the certified plan, then reopen ingress.
        with self.cv:
            self.accepting = False
            while self.pending:
                self.cv.wait()
            self.plan = new_plan
            self.version += 1
            self.accepting = True
            self.cv.notify_all()

    def drain(self) -> None:
        with self.cv:
            while self.pending:
                self.cv.wait()

    def close(self) -> None:
        self.drain()
        for _ in self.threads:
            self.queue.put(None)
        for t in self.threads:
            t.join()

    def _worker(self) -> None:
        while True:
            task = self.queue.get()
            if task is None:
                return
            try:
                got = eval_tree(task.plan, task.matrices)
                ref = left_fold(task.matrices)
                if got != ref:
                    self.errors.append(f"semantic mismatch at seq={task.seq}")
                with self.result_lock:
                    if task.seq in self.results:
                        self.errors.append(f"duplicate seq={task.seq}")
                    self.results[task.seq] = (task.version, got)
            except Exception as exc:  # pragma: no cover - audit path
                self.errors.append(f"seq={task.seq}: {exc!r}")
            finally:
                with self.cv:
                    self.pending -= 1
                    if self.pending == 0:
                        self.cv.notify_all()


def random_matrices(rng: random.Random) -> list[Matrix]:
    dims = (1, 2, 8, 2)
    xs: list[Matrix] = []
    for r, c in zip(dims, dims[1:]):
        xs.append([[rng.randrange(17) for _ in range(c)] for _ in range(r)])
    return xs


def live_cutover_check(tasks: int = 10_000, batch: int = 20) -> dict[str, int]:
    left: Tree = ((0, 1), 2)
    right: Tree = (0, (1, 2))
    runtime = VersionedRuntime(left, workers=4)
    rng = random.Random(0x5EED)
    switches = 0
    for seq in range(tasks):
        runtime.submit(seq, random_matrices(rng))
        if (seq + 1) % batch == 0 and seq + 1 < tasks:
            runtime.reconfigure(right if switches % 2 == 0 else left)
            switches += 1
    runtime.close()
    assert not runtime.errors, runtime.errors[:5]
    assert len(runtime.results) == tasks
    assert set(runtime.results) == set(range(tasks))
    versions = {v for v, _ in runtime.results.values()}
    assert len(versions) == switches + 1
    return {
        "tasks": tasks,
        "cutovers": switches,
        "versions": len(versions),
        "lost": 0,
        "duplicates": 0,
        "semantic_mismatches": 0,
    }


def main() -> int:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "results/independent_runtime.json")
    data = {
        "local_rotation_counterexample": verify_local_trap(),
        "asymptotic_separation": verify_separation(),
        "live_versioned_cutover": live_cutover_check(),
        "status": "all independent Python assertions passed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
