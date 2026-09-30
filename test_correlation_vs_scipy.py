# Cross-check fastdist.correlation against scipy.spatial.distance.correlation
# (weighted and unweighted, centered and not centered, float and int inputs).
#
# Run from the repo root with the `fix-weighted-correlation` branch checked out
# (see test_new_cm_based_metrics.py for the venv setup); exits with 1 if any case
# deviates by more than TOLERANCE.

import sys

import numpy as np
from scipy.spatial import distance

from fastdist import fastdist

SIZE = 10000
TOLERANCE = 1e-12


def main():
	"""compare fastdist.correlation with scipy for all input/parameter combinations and report deviations"""

	rs = np.random.RandomState(seed=0)
	inputs = {
		'float64': (rs.rand(SIZE), rs.rand(SIZE)),
		'int64': (rs.randint(0, 10, SIZE), rs.randint(0, 10, SIZE)),
	}
	w = rs.rand(SIZE)
	cases = [(dtype, weighted, centered)
			 for dtype in inputs for weighted in (True, False) for centered in (True, False)]

	failures = 0
	for i, (dtype, weighted, centered) in enumerate(cases, start=1):
		u, v = inputs[dtype]
		case_w = w if weighted else None
		result_fastdist = fastdist.correlation(u, v, case_w, centered)
		result_scipy = distance.correlation(u, v, case_w, centered=centered)
		diff = abs(result_fastdist - result_scipy)
		status = 'ok' if diff <= TOLERANCE else 'FAIL'
		failures += status == 'FAIL'
		print(f"[{i}/{len(cases)}] {dtype:<8} weighted={weighted!s:<5} centered={centered!s:<5} "
			  f"fastdist={result_fastdist:.15f} scipy={result_scipy:.15f} diff={diff:.1e} {status}")

	print(f"\n{len(cases) - failures} of {len(cases)} cases within {TOLERANCE:.0e} of scipy")
	sys.exit(1 if failures else 0)


if __name__ == '__main__':
	main()
