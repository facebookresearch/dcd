# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os 
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("bright")


"""
Usage:

Plot CarRacingF1 Benchmark results:

python results/plot_f1.py \
-r results/car_racing_f1 \
-f \
f1-dr-5M_steps.csv \
f1-paired-5M_steps.csv \
f1-repaired-5M_steps.csv \
f1-plr-5M_steps.csv \
f1-robust_plr-5M_steps.csv \
-l DR PAIRED REPAIRED PLR "PLR Robust" \
--num_test_episodes 10 \
--threshold 477.71 \
--threshold_label 'Tang et al, 2020' \
--savename f1_eval

"""

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-r', '--result_path',
		type=str,
		default='result/',
		help='Relative path to results directory.'
	)
	parser.add_argument(
		'-f', '--files',
		type=str,
		nargs='+',
		default=[],
		help='Name of results .csv file, output by eval.py.'
	)
	parser.add_argument(
		'-l', '--labels',
		type=str,
		nargs='+',
		default=[],
		help='Name of condition corresponding to each results file.'
	)
	parser.add_argument(
		'-p', '--row_prefix',
		type=str,
		default='test_returns',
		help='Plot rows in results .csv whose metric column matches this prefix.'
	)
	parser.add_argument(
		'-t', '--num_test_episodes',
		type=int,
		default=10
	)
	parser.add_argument(
		'--savename',
		type=str,
		default="latest",
		help='Filename of saved .pdf of plot, saved to figures/.'
	)
	parser.add_argument(
		'--figsize',
		type=str,
		default="2,2",
		help='Dimensions of output figure.'
	)
	parser.add_argument(
		'--threshold',
		type=float,
		default=None
	)
	parser.add_argument(
		'--threshold_label',
		type=str,
		default=None
	)

	return parser.parse_args()


def agg_test_episodes_by_seed(row, num_test_episodes, stat='mean'):
	assert(len(row) % num_test_episodes == 0)

	total_steps = len(row) // num_test_episodes
	row = [float(x) for x in row]
	step = num_test_episodes 

	return [np.mean(row[i*step:i*step + step]) for i in range(total_steps)]


LABEL_COLORS = {
	'DR': 'gray',
	'PAIRED': (0.8859561388376407, 0.5226505841897354, 0.195714831410001),
	'REPAIRED': (0.2038148518479279, 0.6871367484391159, 0.5309645021239799),
	'PLR': (0.9637256003082545, 0.40964669235271706, 0.7430230442501574),
	'PLR Robust': (0.3711152842731098, 0.6174124752499043, 0.9586047646790773),
	'Robust PLR': (0.3711152842731098, 0.6174124752499043, 0.9586047646790773),
}

if __name__ == '__main__':
	args = parse_args()
	plt.rcParams["figure.figsize"] = eval(args.figsize)

	result_path = os.path.expandvars(os.path.expanduser(args.result_path))

	num_labels = len(args.labels)
	colors = sns.husl_palette(num_labels, h=.1)

	colors_ = []
	for l in args.labels:
		if l in LABEL_COLORS:
			colors_.append(LABEL_COLORS[l])
		else:
			colors_.append(colors[i])
	colors = colors_

	x = np.arange(len(args.files))
	width = 0.35
	all_stats = defaultdict(list)
	for i, f in enumerate(args.files):
		fpath = os.path.join(result_path, f)

		with open(fpath, mode='r', newline='') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				if row[0].startswith(args.row_prefix):
					agg_stats = agg_test_episodes_by_seed(row[1:], args.num_test_episodes)

					all_stats[args.labels[i]] += agg_stats

		label_stats = all_stats[args.labels[i]]
		value = np.mean(label_stats)
		err = stats.sem(label_stats)

		label = args.labels[i]
		if label == 'PLR Robust' or label == 'Robust PLR':
			label = r'$\mathregular{PLR^{\perp}}$'

		x1 = plt.bar(x[i]*width, value, width, label=label, color=colors[i], yerr=err)

	x = np.arange(len(args.files))

	plt.grid()
	sns.despine(top=True, right=True, left=False, bottom=False)
	plt.gca().get_xaxis().set_visible(False)
	plt.ylabel('Test return', fontsize=12)

	# Add threshold
	if args.threshold:
		plt.axhline(y=args.threshold, label=args.threshold_label, color='green', linestyle='dotted')

	legend = plt.legend(
			 ncol=1,
			 loc='upper left', bbox_to_anchor=(1, 1),
			 frameon=False,
			 fontsize=12)

	plt.savefig(f"figures/{args.savename}.pdf", bbox_extra_artists=(legend,), bbox_inches='tight')

	plt.show()
