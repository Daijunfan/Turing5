.PHONY: test audit counterexample core scale exhaustive

test:
	PYTHONPATH=. python3 scripts/run_tests.py

audit:
	PYTHONPATH=. python3 scripts/audit_results.py

counterexample:
	PYTHONPATH=. python3 scripts/reproduce_counterexample.py

core: counterexample test
	PYTHONPATH=. python3 scripts/run_core_experiments.py

scale:
	PYTHONPATH=. python3 scripts/run_scale_experiments.py

exhaustive:
	PYTHONPATH=. python3 scripts/run_small_exhaustive.py --min-n 2 --max-n 4 --workers 8
	PYTHONPATH=. python3 scripts/run_small_exhaustive.py --min-n 5 --max-n 5 --workers 8
	PYTHONPATH=. python3 scripts/run_real_small_exhaustive.py --max-n 5 --workers 8
