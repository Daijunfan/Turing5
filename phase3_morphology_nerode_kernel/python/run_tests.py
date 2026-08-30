#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
from time import perf_counter


MODULES = (
    "tests.test_automaton",
    "tests.test_matrix_chain",
    "tests.test_pricing",
    "tests.test_dag",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    tests = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        tests.extend(
            (f"{module_name}.{name}", function)
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        )
    failures = []
    started = perf_counter()
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:
            failures.append((name, str(error)))
            print(f"FAIL {name}: {error}", file=sys.stderr)
    print(f"tests={len(tests)} failures={len(failures)} seconds={perf_counter()-started:.3f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
