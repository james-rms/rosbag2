import argparse
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas
from pathlib import Path

LABEL_CATEGORIES = ["storage_config", "cache_size"]
OUTPUT_METRIC = "total_recorded_count"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="input CSV to draw plots of")
    parser.add_argument("--outdir", default="./plots", help="output directory to write plots to")
    parser.add_argument("--title", help="A title to add to each plot")
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    frame = pandas.read_csv(args.csv)
    means = frame.groupby(LABEL_CATEGORIES)[OUTPUT_METRIC].mean().rename("mean")
    # multiply by 3.92 to turn a std error to 95% confidence interval
    confidence_intervals = frame.groupby(LABEL_CATEGORIES)[OUTPUT_METRIC].std().rename("ci") * 3.92
    joined = pandas.concat([means, confidence_intervals], axis=1).sort_values("mean", ascending=False)


    cache_size_labels = joined.index.get_level_values(1).unique()
    for cache_size_label in cache_size_labels:
        series = joined[
            joined.index.isin([cache_size_label], level=1)
        ]
        plt.bar(
            x=series.index.get_level_values(0),
            height=series["mean"],
            yerr=series["ci"],
            align='edge',
            ecolor='black',
            color=["skyblue" if "mcap" in name else "aquamarine" for name in series.index.get_level_values(0)],
            capsize=10,
        )
        plt.ylabel("Messages Recorded (higher is better)")
        title = f"cache size: {cache_size_label}"
        if args.title is not None:
            title = f"{args.title}\n{title}"
        plt.title(title)
        plt.tick_params(labelrotation=45)
        plt.tight_layout()
        outpath = Path(args.outdir) / f"{cache_size_label}.png"
        print(f"saving figure to {outpath}", file=sys.stderr)
        plt.savefig(outpath)
        plt.clf()
        high = max(series["mean"])
        low = min(series["mean"])
        plt.ylim([low-0.5*(high-low), high+0.5*(high-low)])


if __name__ == "__main__":
    main()
