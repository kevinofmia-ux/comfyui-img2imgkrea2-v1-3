"""Dependency-free test runner.

    python tests/run_tests.py            # everything
    python tests/run_tests.py pipeline   # only modules whose name contains this

Deliberately not pytest-dependent: this has to run inside whatever Python
environment ComfyUI happens to be installed into, and that environment is not
guaranteed to have a test framework. ``pytest tests`` works too.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "tests.test_colorspace",
    "tests.test_transport",
    "tests.test_lut",
    "tests.test_pipeline",
    "tests.test_edge_cases",
    "tests.test_node",
    "tests.test_quality",
]


def main(argv: list[str]) -> int:
    selector = argv[0] if argv else ""
    modules = [m for m in MODULES if selector in m]

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()

    for name in modules:
        module = importlib.import_module(name)
        tests = sorted(n for n in dir(module) if n.startswith("test_"))
        print(f"\n{name}  ({len(tests)} tests)")
        for test_name in tests:
            fn = getattr(module, test_name)
            t0 = time.perf_counter()
            try:
                result = fn()
            except Exception:
                failed += 1
                failures.append((f"{name}.{test_name}", traceback.format_exc()))
                print(f"  FAIL  {test_name}")
                continue
            elapsed = (time.perf_counter() - t0) * 1000.0
            if isinstance(result, str) and result.startswith("skipped"):
                skipped += 1
                print(f"  SKIP  {test_name}  ({result})")
            else:
                passed += 1
                print(f"  ok    {test_name}  ({elapsed:.0f} ms)")

    for name, tb in failures:
        print(f"\n{'=' * 70}\nFAILED {name}\n{'=' * 70}\n{tb}")

    total = time.perf_counter() - started
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped in {total:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
