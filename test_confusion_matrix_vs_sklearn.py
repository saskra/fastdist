# Cross-check fastdist.confusion_matrix against sklearn.metrics.confusion_matrix
# (dtypes, label sets incl. negative/large labels, weights, given labels, normalize).
#
# Run from the repo root with the `fix-confusion-matrix-labels` branch checked out
# (see test_new_cm_based_metrics.py for the venv setup); exits with 1 if any check
# fails. Known failures that also occur on master: mixed int64/uint8 inputs raise a
# numba TypingError, and normalize='pred' gives nan where sklearn gives 0 for a
# never-predicted class.

import sys
import warnings

import numpy as np
from sklearn import metrics

from fastdist import fastdist


def main():
	"""compare fastdist.confusion_matrix with sklearn over dtypes, label sets, weights, labels and normalize"""
	warnings.simplefilter('ignore', RuntimeWarning)  # 0/0 in normalize for empty rows, same in both
	rs = np.random.RandomState(0)
	n = 5000
	w = rs.rand(n)
	cases = {
		'binary int64': (rs.randint(0, 2, n), rs.randint(0, 2, n)),
		'binary uint8': (rs.randint(0, 2, n).astype(np.uint8), rs.randint(0, 2, n).astype(np.uint8)),
		'5 classes': (rs.randint(0, 5, n), rs.randint(0, 5, n)),
		'labels 2..3': (rs.randint(2, 4, n), rs.randint(2, 4, n)),
		'negative/large': (rs.choice([-5, 3, 1000, 70000], n), rs.choice([-5, 3, 1000, 70000], n)),
		'pred misses class': (rs.randint(0, 3, n), rs.randint(0, 2, n)),
		'mixed int64/uint8': (rs.randint(0, 3, n), rs.randint(0, 3, n).astype(np.uint8)),
		'20 classes': (rs.randint(0, 20, n), rs.randint(0, 20, n)),
	}
	failures, total, errors = 0, 0, 0
	for i, (name, (t, p)) in enumerate(cases.items(), start=1):
		lab = np.unique(np.concatenate((t, p)))[::-1].astype(t.dtype)
		checks = {
			'plain': (lambda: fastdist.confusion_matrix(t, p), metrics.confusion_matrix(t, p)),
			'weighted': (lambda: fastdist.confusion_matrix(t, p, None, w), metrics.confusion_matrix(t, p, sample_weight=w)),
			'labels': (lambda: fastdist.confusion_matrix(t, p, lab), metrics.confusion_matrix(t, p, labels=lab)),
		}
		for norm in ('true', 'pred', 'all'):
			checks['norm=' + norm] = (lambda norm=norm: fastdist.confusion_matrix(t, p, None, None, norm),
									  metrics.confusion_matrix(t, p, normalize=norm))
		results = []
		for key, (fn, ref) in checks.items():
			total += 1
			try:
				ok = np.allclose(fn(), ref, equal_nan=True)
			except Exception as e:
				ok, errors = False, errors + 1
				key += ' (' + type(e).__name__ + ')'
			failures += not ok
			results.append(key + ('' if ok else ' FAIL'))
		failed = [r for r in results if 'FAIL' in r]
		print(f"[{i}/{len(cases)}] {name:<18} " + (', '.join(failed) if failed else 'all ok'))
	print(f"\n{total - failures} of {total} checks match sklearn ({errors} raised)")
	sys.exit(1 if failures else 0)


if __name__ == '__main__':
	main()
