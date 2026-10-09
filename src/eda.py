"""Phase 2: exploratory data analysis.

    python -m src.eda --csv data/malicious_phish.csv

Writes figures and a summary table to reports/. This becomes the dataset
characteristics subsection of the paper, and it is worth doing before modelling
rather than after, because it is where you find out whether the classes are
separable on simple features at all.
"""

import argparse
from collections import Counter
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                       # noqa: E402

from . import config, data, features      # noqa: E402


def run(csv, max_rows=0):
    config.ensure_dirs()
    df = data.load(csv, max_rows)
    df["klass"] = df["y"].map({0: "benign", 1: "malicious"})

    feats = pd.DataFrame(features.extract_batch(df["url"]),
                         columns=features.FEATURE_NAMES)
    feats["klass"] = df["klass"].values

    summary = feats.groupby("klass").mean().T
    summary["difference"] = (summary["malicious"] - summary["benign"]).abs()
    summary = summary.sort_values("difference", ascending=False)
    summary.to_csv(config.REPORTS / "eda_feature_summary.csv")
    print("\nfeatures that separate the classes most:")
    print(summary.head(12).round(3).to_string())

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    for k, sub in feats.groupby("klass"):
        ax[0, 0].hist(sub["url_length"].clip(0, 200), bins=50, alpha=0.6, label=k)
    ax[0, 0].set_title("URL length")
    ax[0, 0].set_xlabel("characters")
    ax[0, 0].legend()

    for k, sub in feats.groupby("klass"):
        ax[0, 1].hist(sub["url_entropy"], bins=50, alpha=0.6, label=k)
    ax[0, 1].set_title("Character entropy")
    ax[0, 1].legend()

    flags = ["has_ip", "is_shortened", "has_https", "has_port",
             "suspicious_word_count", "has_double_slash_in_path"]
    rates = feats.groupby("klass")[flags].mean().T
    rates.plot(kind="barh", ax=ax[1, 0])
    ax[1, 0].set_title("Rate of structural flags")

    tlds = Counter()
    for url, k in zip(df["url"], df["klass"]):
        host = urlparse(url if "://" in url else "http://" + url).hostname or ""
        if "." in host:
            tlds[(host.rsplit(".", 1)[-1], k)] += 1
    top = [t for t, _ in Counter(
        {t: c for (t, k), c in tlds.items()}).most_common(10)]
    tld_df = pd.DataFrame(
        {k: [tlds.get((t, k), 0) for t in top] for k in ("benign", "malicious")},
        index=top)
    tld_df.plot(kind="bar", ax=ax[1, 1])
    ax[1, 1].set_title("Top-level domains")
    ax[1, 1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out = config.REPORTS / "eda_overview.png"
    fig.savefig(out, dpi=150)
    print("\nfigure ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--max-rows", type=int, default=0)
    a = p.parse_args()
    run(a.csv, a.max_rows)
