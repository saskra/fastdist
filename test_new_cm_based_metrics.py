# Author: Dr. Sascha D. Krauss
# Contact: sascha.krauss@uk-essen.de
# Start date: 2022-02-17

import json
import os
import platform
import sys
import timeit
from datetime import datetime, timezone

import numba
import numpy as np
import sklearn
from sklearn import metrics

from fastdist import fastdist
from fastdist import __version__ as fastdist_version

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def get_mpv(cm):
	"""calculate and return quality metrics based on the diagonal of the confusion matrix

	Args:
		cm: confusion matrix
	"""

	n_classes = cm.shape[0]
	cm_col = cm / cm.sum(axis=0)[np.newaxis, :]
	mpv = np.trace(cm_col) / n_classes  # Mean predictive value

	return mpv


def get_miou(cm):
	"""calculate and return quality metrics based on the diagonal of the confusion matrix

	Args:
		cm: confusion matrix
	"""

	n_classes = cm.shape[0]

	# calculate the necessary matrix for the mean Intersection over Union (IoU)
	cm_iou = np.zeros((n_classes, n_classes))
	for row in range(n_classes):
		cm_row_sum = cm[row, :].sum()
		for col in range(n_classes):
			# tp / (tp + fp + fn)
			cm_iou[row, col] = cm[row, col] / (cm_row_sum + cm[:, col].sum() - cm[row, col])

	miou = np.trace(cm_iou) / n_classes  # Mean IoU/ Jaccard

	return miou


def record(results, size, metric, time_primary, time_without_cm=None, reference_label=None, time_reference=None):
	"""append one benchmark row to the results list

	Args:
		results: list to append the row to
		size: vector length this row was measured at
		metric: name of the benchmarked fastdist function
		time_primary: time of the fastdist call this row is about (the precomputed-cm variant, where applicable)
		time_without_cm: time of the equivalent fastdist call without a precomputed cm, if measured
		reference_label: name of the external reference this was compared against ("sklearn" or "own_function")
		time_reference: time of the reference call, if measured
	"""

	results.append({
		'size': size,
		'metric': metric,
		'time_primary': time_primary,
		'time_without_cm': time_without_cm,
		'speedup_cm': (time_without_cm / time_primary) if time_without_cm else None,
		'reference_label': reference_label,
		'time_reference': time_reference,
		'speedup_reference': (time_reference / time_primary) if time_reference else None,
	})


def print_summary(results):
	"""print a compact table of all benchmark rows

	Args:
		results: list of benchmark rows as produced by record()
	"""

	header = f"{'Metric':<22}{'Size':>12}{'Fastdist (s)':>14}{'Speedup cm':>12}{'Reference':>15}{'Speedup ref':>13}"
	print()
	print(header)
	print('-' * len(header))
	for row in results:
		speedup_cm = f"{row['speedup_cm']:.1f}x" if row['speedup_cm'] else '-'
		reference = row['reference_label'] or '-'
		speedup_reference = f"{row['speedup_reference']:.1f}x" if row['speedup_reference'] else '-'
		print(f"{row['metric']:<22}{row['size']:>12}{row['time_primary']:>14.6f}"
			  f"{speedup_cm:>12}{reference:>15}{speedup_reference:>13}")


def save_results(results):
	"""save the benchmark results together with a run log for reproducibility

	Args:
		results: list of benchmark rows as produced by record()
	"""

	os.makedirs(RESULTS_DIR, exist_ok=True)
	timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

	results_path = os.path.join(RESULTS_DIR, f'benchmark_{timestamp}.json')
	with open(results_path, 'w') as f:
		json.dump(results, f, indent=2)

	run_log = {
		'timestamp': timestamp,
		'sizes': sorted({row['size'] for row in results}),
		'versions': {
			'python': sys.version,
			'fastdist': fastdist_version,
			'numba': numba.__version__,
			'numpy': np.__version__,
			'scikit-learn': sklearn.__version__,
		},
		'platform': platform.platform(),
	}
	run_log_path = os.path.join(RESULTS_DIR, f'run_log_{timestamp}.json')
	with open(run_log_path, 'w') as f:
		json.dump(run_log, f, indent=2)

	print(f"\nSaved results to {results_path}")
	print(f"Saved run log to {run_log_path}")


def main():
	"""benchmark and cross-check the confusion-matrix-based fastdist metrics against sklearn and reference formulas"""

	sizes = [10000, 100000000]
	results = []

	for size in sizes:
		print('Vector length: ' + str(size))
		y_true = np.random.randint(3, size=size)
		y_pred = np.random.randint(3, size=size)

		print('Confusion matrix')
		start = timeit.default_timer()
		cm1 = fastdist.confusion_matrix(y_true, y_pred)
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time fastdist: ', duration1)
		cm2 = metrics.confusion_matrix(y_true, y_pred)
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time sklearn: ', duration2)
		print('Fastdist took '
			  + str(duration1 / duration2)
			  + ' times as much time as sklearn.')
		record(results, size, 'confusion_matrix', duration1, reference_label='sklearn', time_reference=duration2)

		print('Overall Accuracy')
		start = timeit.default_timer()
		acc1 = fastdist.accuracy_score(y_true, y_pred, cm1)
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		acc2 = fastdist.accuracy_score(y_true, y_pred)
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist with precalculated confusion matrix took '
			  + str(duration1 / duration2)
			  + ' times as much time as without.')
		assert acc1 == acc2
		start = timeit.default_timer()
		acc3 = metrics.accuracy_score(y_true, y_pred)
		stop = timeit.default_timer()
		duration3 = stop - start
		print('Time: ', duration3)
		print('Fastdist took '
			  + str(duration1 / duration3)
			  + ' times as much time as sklearn.')
		assert acc2 == acc3
		record(results, size, 'accuracy_score', duration1, time_without_cm=duration2,
			   reference_label='sklearn', time_reference=duration3)

		print('Mean predictive value')
		start = timeit.default_timer()
		mpv1 = fastdist.mean_predictive_value(y_true, y_pred, cm1)
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		mpv2 = get_mpv(cm1)
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist  took '
			  + str(duration1 / duration2)
			  + ' times as much time as own function.')
		assert mpv1 == mpv2
		record(results, size, 'mean_predictive_value', duration1, reference_label='own_function',
			   time_reference=duration2)

		print('Mean Intersection over Union')
		start = timeit.default_timer()
		miou1 = fastdist.mean_iou(y_true, y_pred, cm1)
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		miou2 = get_miou(cm1)
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist  took '
			  + str(duration1 / duration2)
			  + ' times as much time as own function.')
		assert miou1 == miou2
		record(results, size, 'mean_iou', duration1, reference_label='own_function', time_reference=duration2)

		print('Balanced Accuracy')
		start = timeit.default_timer()
		bal1 = fastdist.balanced_accuracy_score(y_true, y_pred, cm1)
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		bal2 = fastdist.balanced_accuracy_score(y_true, y_pred)
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist with precalculated confusion matrix took '
			  + str(duration1 / duration2)
			  + ' times as much time as without.')
		assert bal1 == bal2
		start = timeit.default_timer()
		bal3 = metrics.balanced_accuracy_score(y_true, y_pred)
		stop = timeit.default_timer()
		duration3 = stop - start
		print('Time: ', duration3)
		print('Fastdist took '
			  + str(duration1 / duration3)
			  + ' times as much time as sklearn.')
		assert abs(bal2 - bal3) < 1e-9
		record(results, size, 'balanced_accuracy_score', duration1, time_without_cm=duration2,
			   reference_label='sklearn', time_reference=duration3)

		print('Precision (macro)')
		cm_pred = fastdist.confusion_matrix(y_true, y_pred, normalize='pred')
		start = timeit.default_timer()
		prec1 = fastdist.precision_score(y_true, y_pred, cm_pred, average='macro')
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		prec2 = fastdist.precision_score(y_true, y_pred, average='macro')
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist with precalculated confusion matrix took '
			  + str(duration1 / duration2)
			  + ' times as much time as without.')
		assert (prec1 == prec2).all()
		start = timeit.default_timer()
		prec3 = metrics.precision_score(y_true, y_pred, average='macro')
		stop = timeit.default_timer()
		duration3 = stop - start
		print('Time: ', duration3)
		print('Fastdist took '
			  + str(duration1 / duration3)
			  + ' times as much time as sklearn.')
		assert abs(prec2[0] - prec3) < 1e-9
		record(results, size, 'precision_score (macro)', duration1, time_without_cm=duration2,
			   reference_label='sklearn', time_reference=duration3)

		print('Recall (macro)')
		cm_true = fastdist.confusion_matrix(y_true, y_pred, normalize='true')
		start = timeit.default_timer()
		rec1 = fastdist.recall_score(y_true, y_pred, cm_true, average='macro')
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		rec2 = fastdist.recall_score(y_true, y_pred, average='macro')
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist with precalculated confusion matrix took '
			  + str(duration1 / duration2)
			  + ' times as much time as without.')
		assert (rec1 == rec2).all()
		start = timeit.default_timer()
		rec3 = metrics.recall_score(y_true, y_pred, average='macro')
		stop = timeit.default_timer()
		duration3 = stop - start
		print('Time: ', duration3)
		print('Fastdist took '
			  + str(duration1 / duration3)
			  + ' times as much time as sklearn.')
		assert abs(rec2[0] - rec3) < 1e-9
		record(results, size, 'recall_score (macro)', duration1, time_without_cm=duration2,
			   reference_label='sklearn', time_reference=duration3)

		print('F1 (micro)')
		start = timeit.default_timer()
		f1_1 = fastdist.f1_score(y_true, y_pred, cm1, average='micro')
		stop = timeit.default_timer()
		duration1 = stop - start
		print('Time: ', duration1)
		start = timeit.default_timer()
		f1_2 = fastdist.f1_score(y_true, y_pred, average='micro')
		stop = timeit.default_timer()
		duration2 = stop - start
		print('Time: ', duration2)
		print('Fastdist with precalculated confusion matrix took '
			  + str(duration1 / duration2)
			  + ' times as much time as without.')
		assert (f1_1 == f1_2).all()
		start = timeit.default_timer()
		f1_3 = metrics.f1_score(y_true, y_pred, average='micro')
		stop = timeit.default_timer()
		duration3 = stop - start
		print('Time: ', duration3)
		print('Fastdist took '
			  + str(duration1 / duration3)
			  + ' times as much time as sklearn.')
		assert abs(f1_2[0] - f1_3) < 1e-9
		record(results, size, 'f1_score (micro)', duration1, time_without_cm=duration2,
			   reference_label='sklearn', time_reference=duration3)

	print_summary(results)
	save_results(results)


if __name__ == '__main__':
	main()
