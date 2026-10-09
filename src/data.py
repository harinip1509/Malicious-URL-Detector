"""Phase 1: load the dataset, clean it, and split it honestly.

The cleaning here is not cosmetic. This dataset contains exact duplicates and,
more importantly, URLs that appear under both a benign and a malicious label.
Leave those in and the same string lands in train and test, which inflates every
score you report. Removing them is the difference between a result you can
defend and one you cannot.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load(csv_path: str, max_rows: int = 0, seed: int = 42, verbose: bool = True):
    df = pd.read_csv(csv_path)

    url_col = next((c for c in df.columns if c.lower() in ("url", "urls")), None)
    lab_col = next((c for c in df.columns
                    if c.lower() in ("type", "label", "class", "result", "status")), None)
    if url_col is None or lab_col is None:
        raise ValueError(f"Could not find url and label columns in {list(df.columns)}")

    df = df[[url_col, lab_col]].rename(columns={url_col: "url", lab_col: "label"})
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 4]

    df["y"] = (~df["label"].astype(str).str.lower()
               .isin(config.BENIGN_LABELS)).astype(int)

    # Order matters here. Deduplicating first would delete the second copy of a
    # URL labelled both ways, so the conflict would never be detected and the
    # surviving row could land in the wrong split.
    conflicting = df.groupby("url")["y"].nunique()
    leaked = set(conflicting[conflicting > 1].index)
    if leaked:
        df = df[~df["url"].isin(leaked)]

    n0 = len(df)
    df = df.drop_duplicates(subset="url")
    dupes = n0 - len(df)

    if max_rows and len(df) > max_rows:
        frac = max_rows / len(df)
        # Sample classes independently. Besides preserving the stratified
        # balance, this keeps the grouping column on pandas versions where
        # DataFrameGroupBy.apply omits grouping columns from its input.
        df = pd.concat(
            [g.sample(max(int(len(g) * frac), 1), random_state=seed)
             for _, g in df.groupby("y")],
            ignore_index=True,
        )

    df = df.reset_index(drop=True)
    if verbose:
        print(f"rows {len(df)}   removed {dupes} duplicates, {len(leaked)} conflicting labels")
        print(f"class balance: benign {(df.y == 0).sum()}  malicious {(df.y == 1).sum()} "
              f"({df.y.mean():.1%} malicious)")
        print("original label counts:\n", df["label"].value_counts().to_string())
    return df


def split(df, cfg: config.TrainConfig, verbose: bool = True):
    hold = cfg.val_size + cfg.test_size
    train, tmp = train_test_split(df, test_size=hold, stratify=df["y"],
                                  random_state=cfg.seed)
    val, test = train_test_split(tmp, test_size=cfg.test_size / hold,
                                 stratify=tmp["y"], random_state=cfg.seed)
    if verbose:
        print(f"train {len(train)}   val {len(val)}   test {len(test)}")
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))
