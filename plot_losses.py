#!/usr/bin/env python3
"""Parse a CUT/pix2pix loss_log.txt and plot smoothed loss curves -> PNG.

Usage: python3 plot_losses.py <loss_log.txt> [out.png] [title]
"""
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"\(epoch: (\d+), iters: (\d+), time: [\d.]+, data: [\d.]+\) (.*)")
KV = re.compile(r"(\w+): ([-\d.eE]+)")


def parse(path):
    series = defaultdict(list)
    xs = []
    with open(path) as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            ep = int(m.group(1))
            kvs = dict((k, float(v)) for k, v in KV.findall(m.group(3)))
            xs.append(ep + 0.0)
            for k, v in kvs.items():
                series[k].append(v)
    return xs, series


def smooth(y, w=15):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "losses.png"
    title = sys.argv[3] if len(sys.argv) > 3 else path
    xs, series = parse(path)
    if not xs:
        print("no loss lines parsed"); return
    keys = [k for k in series if any(t in k for t in ("G_", "D_"))]
    fig, ax = plt.subplots(figsize=(11, 6))
    for k in sorted(keys):
        y = np.array(series[k])
        ax.plot(xs[:len(y)], smooth(y), label=k, lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (smoothed)")
    ax.set_title(title); ax.legend(ncol=3, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("wrote", out)
    # also print last-epoch means
    last_ep = int(max(xs))
    print(f"final epoch {last_ep} mean losses:")
    for k in sorted(keys):
        y = [v for x, v in zip(xs, series[k]) if int(x) == last_ep]
        if y:
            print(f"  {k:8s}: {np.mean(y):.4f}")


if __name__ == "__main__":
    main()
