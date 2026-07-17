# Reproduce all results and figures.
# Usage:  make all        (benchmarks + all figures, ~10 min single-core)
#         make figures    (regenerate figures from committed .npz data)

PY ?= python3

all: benchmarks figures

benchmarks:
	cd code && $(PY) bench1_velocity.py
	cd code && $(PY) bench3_final.py

figures:
	cd code && $(PY) make_val_figs.py
	cd code && $(PY) figures_corrected.py

diagnostics:
	cd code && $(PY) test1_flow.py
	cd code && $(PY) test2_decompose.py
	cd code && $(PY) test3_whichU1.py

.PHONY: all benchmarks figures diagnostics
