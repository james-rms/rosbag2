import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas
from pathlib import Path

LABEL_CATEGORIES = ["plugin_config", "messages", "batch_size"]
OUTPUT_METRIC = "avg_byte_throughput"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="input CSV to draw plots of")
    parser.add_argument("--color", default="skyblue", help="colour of bars to plot")
    parser.add_argument("--outdir", default="./plots", help="output directory to write plots to")
    parser.add_argument("--title", help="A title to add to each plot")
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    frame = pandas.read_csv(args.csv)
    means = frame.groupby(LABEL_CATEGORIES)[OUTPUT_METRIC].mean().rename("mean")
    # multiply by 3.92 to turn a std error to 95% confidence interval
    confidence_intervals = frame.groupby(LABEL_CATEGORIES)[OUTPUT_METRIC].std().rename("ci") * 3.92
    joined = pandas.concat([means, confidence_intervals], axis=1).sort_values("mean", ascending=False)


    message_labels = joined.index.get_level_values(1).unique()
    batch_labels = joined.index.get_level_values(2).unique()
    for message_label in message_labels:
        for batch_label in batch_labels:
            series = joined[
                np.logical_and(
                    joined.index.isin([message_label], level=1),
                    joined.index.isin([batch_label], level=2)
                )
            ]
            plt.bar(
                x=series.index.get_level_values(0),
                height=series["mean"],
                yerr=series["ci"],
                align='edge',
                ecolor='black',
                color=args.color,
                capsize=10,
            )
            plt.ylabel("Throughput (bytes/s)")
            plt.title(f"Throughput: batch size: {batch_label}, message sizes: {message_label}")
            plt.tick_params(labelrotation=45)
            if args.title is not None:
                plt.title(args.title)
            plt.tight_layout()
            outpath = Path(args.outdir) / f"{message_label}_{batch_label}.png"
            print(f"saving figure to {outpath}", file=sys.stderr)
            plt.savefig(outpath)
            plt.clf()


if __name__ == "__main__":
    main()
