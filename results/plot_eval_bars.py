# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("bright")

"""
Usage:

Plot MiniGrid results for 25 blocks 
(Robust PLR and REPAIRED uses half as many gradients as baselines):

python results/plot_eval_bars.py \
-r results/minigrid_ood \
-f \
mg_25_blocks-dr-250M_steps.csv \
mg_25_blocks-minimax-250M_steps.csv \
mg_25_blocks-paired-250M_steps.csv \
mg_25_blocks-repaired-250M_steps.csv \
mg_25_blocks-plr-250M_steps.csv \
mg_25_blocks-robust_plr-250M_steps.csv \
-l "DR" Minimax PAIRED REPAIRED PLR "Robust PLR" \
--savename minigrid_25_blocks_eval

------------------------------------------------------------------------
Plot MiniGrid results for uniform block count in [0,60]:

python results/plot_eval_bars.py \
-r results/minigrid_ood \
-f \
mg_60_blocks_uni-dr_20k_updates.csv \
mg_60_blocks_uni-robust_plr_20k_updates.csv \
mg_60_blocks-accel_20k_updates.csv \
-l "DR" "Robust PLR" ACCEL \
--figsize 24,2 \
--savename minigrid_60_blocks_uni_eval


Plot BipedalWalker results:

python results/plot_eval_bars.py \
-r results/bipedal \
-f \
bipedal8d-dr_20k-updates.csv \
bipedal8d-robust_plr-20k_updates.csv \
bipedal8d-accel_20k-updates.csv \
-l "DR" "Robust PLR" ACCEL \
--savename bipedal_eval

"""


def parse_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "-r",
        "--result_path",
        type=str,
        default="results/",
        help="Relative path to results directory.",
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        default=["test.csv", "test2.csv"],
        help="Name of results .csv file, output by eval.py.",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        default=[],
        help="Name of condition corresponding to each results file.",
    )
    parser.add_argument(
        "--row_prefix",
        type=str,
        default="solved_rate",
        help="Plot rows in results .csv whose metric column matches this prefix.",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        nargs="+",
        default=[],
        help="List of metric names to plot, without the --row_prefix.",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=None,
        help="Further filter matched metric rows with a list of substrings.",
    )
    parser.add_argument(
        "--ylabel", type=str, default="Solved rate", help="Y-axis label."
    )
    parser.add_argument(
        "--savename",
        type=str,
        default="latest",
        help="Filename of saved .pdf of plot, saved to figures/.",
    )
    parser.add_argument(
        "--figsize", type=str, default="(14,2)", help="Dimensions of output figure."
    )

    return parser.parse_args()


LABEL_COLORS = {
    "DR": "gray",
    "Minimax": "red",
    "PAIRED": (0.8859561388376407, 0.5226505841897354, 0.195714831410001),
    "REPAIRED": (0.2038148518479279, 0.6871367484391159, 0.5309645021239799),
    "PLR": (0.9637256003082545, 0.40964669235271706, 0.7430230442501574),
    "PLR Robust": (0.3711152842731098, 0.6174124752499043, 0.9586047646790773),
    "Robust PLR": (0.3711152842731098, 0.6174124752499043, 0.9586047646790773),
    "ACCEL": (0.30588235, 0.83921569, 0.27843137),
}

ENV_ALIASES = {
    "SixteenRooms": "16Rooms",
    "SixteenRoomsFewerDoors": "16Rooms2",
    "SimpleCrossingS11N5": "SimpleCrossing",
    "PerfectMazeMedium": "PerfectMazeMed",
    "BipedalWalkerHardcore": "HardCore",
}


if __name__ == "__main__":
    args = parse_args()

    assert len(args.files) == len(args.labels)

    num_labels = len(args.labels)
    colors = sns.husl_palette(num_labels, h=0.1)

    df = pd.DataFrame()
    for i, f in enumerate(args.files):
        fpath = os.path.join(args.result_path, f)
        df_ = pd.read_csv(fpath)

        df_ = df_[df_["metric"].str.startswith(args.row_prefix)]

        if args.include is not None:
            df_ = df_[df_["metric"].str.contains("|".join(args.include))]

        df_["label"] = args.labels[i]

        out_cols = ["metric", "label"]

        df_["median"] = df_.median(axis=1)
        df_["q1"] = df_.quantile(0.25, axis=1)
        df_["q3"] = df_.quantile(0.75, axis=1)
        out_cols += ["median", "q1", "q3"]

        out = df_[out_cols]
        df = pd.concat([df, out])

    df_metrics = df["metric"].unique()
    num_subplots = len(df_metrics)
    nrows, ncols = 1, num_subplots
    f, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharey=True, sharex=True, figsize=eval(args.figsize)
    )

    x = np.arange(len(args.labels))
    width = 0.35

    for i, ax in enumerate(axes.flatten()):
        metric = df_metrics[i]

        for j, label in enumerate(args.labels):
            idx = (df["metric"] == metric) & (df["label"] == label)

            if label in LABEL_COLORS:
                color = LABEL_COLORS[label]
            else:
                color = colors[j]

            if label == "PLR Robust" or label == "Robust PLR":
                label = r"$\mathregular{PLR^{\perp}}$"

            value = df[idx]["median"]
            error = (df[idx]["q3"] - df[idx]["q1"]) / 2.0

            x1 = ax.bar(
                x[j] * width, value, width, label=label, color=color, yerr=error
            )

        title = metric.split(":")[-1].split("-")
        if len(title) > 2:
            title = "".join(title[1:-1])
        elif len(title) > 1:
            title = title[0]
        if title in ENV_ALIASES:
            title = ENV_ALIASES[title]

        ax.set_title(title, fontsize=14)
        ax.grid()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(0, 1)

        if i == 0:
            ax.set_ylabel(args.ylabel, fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    legend = f.legend(
        handles,
        labels,
        ncol=len(args.labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        frameon=False,
        fontsize=12,
    )

    plt.savefig(
        f"figures/{args.savename}.pdf",
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    plt.show()
