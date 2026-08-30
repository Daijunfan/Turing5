#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect
import sys
from time import perf_counter


MODULES = (
    "tests.test_counterexample",
    "tests.test_residual_kernel",
    "tests.test_real_migration",
    "tests.test_independent_oracles",
)


def main() -> None:
    started = perf_counter()
    tests = []
    for name in MODULES:
        module = importlib.import_module(name)
        tests.extend(
            (f"{name}.{member_name}", member)
            for member_name, member in inspect.getmembers(module, inspect.isfunction)
            if member_name.startswith("test_")
        )
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as error:  # deliberate minimal standalone test harness
            failures.append((name, error))
            print(f"FAIL {name}: {error}", file=sys.stderr)
    print(
        f"tests={len(tests)} failures={len(failures)} seconds={perf_counter()-started:.3f}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

