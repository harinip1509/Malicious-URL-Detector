"""Produce every figure, table and number the paper needs.

Run after src/train.py has written artifacts/ and reports/:

    python -m src.paper_figures

Outputs in reports/:
    fig1_training_curve.png   (only if train.py logged losses)
    fig2_roc_curves.png       ROC for each base learner and the proposed model
    fig3_confusion_matrix.png confusion matrix of the proposed model
    fig4_adversarial.png      recall under each evasion attack
    paper_tables.txt          Tables I, III and IV already formatted for pasting
    paper_numbers.txt         abstract figures, latency, dataset counts

Nothing here invents values. Every number is computed from the saved models and
the held-out test split, so what the paper reports is what the run produced.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from sklearn.metrics import (auc, classification_report, confusion_matrix,  # noqa: E402
                             roc_curve)

from . import adversarial, config, ensemble   # noqa: E402
from .predictor import Predictor              # noqa: E402

PRETTY = {"bert": "DistilBERT", "xgboost": "XGBoost", "lightgbm": "LightGBM",
          "catboost": "CatBoost", "adaboost": "AdaBoost"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


def load_test():
    path = config.REPORTS / "test_split.csv"
    if not path.exists():
        raise SystemExit("reports/test_split.csv missing. Run python -m src.train first.")
    df = pd.read_csv(path)
    return df["url"].astype(str).tolist(), df["y"].values


def batched_probs(pred, urls, batch=512):
    out = {k: [] for k in config.BASE_ORDER}
    for i in range(0, len(urls), batch):
        chunk = urls[i:i + batch]
        p = pred.base_probabilities(chunk)
        for k in config.BASE_ORDER:
            out[k].append(np.asarray(p[k]))
        print(f"  scored {min(i + batch, len(urls))}/{len(urls)}", end="\r")
    print()
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    config.ensure_dirs()
    pred = Predictor()
    urls, y = load_test()
    print(f"test set: {len(urls)} URLs")

    probs = batched_probs(pred, urls)
    final = ensemble.apply_meta(pred.meta, probs)

    # ---------------------------------------------------------------- Fig 2, ROC
    plt.figure(figsize=(3.4, 2.8))
    for k in config.BASE_ORDER:
        fpr, tpr, _ = roc_curve(y, probs[k])
        plt.plot(fpr, tpr, lw=0.9, alpha=0.75, label=f"{PRETTY[k]} ({auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(y, final)
    plt.plot(fpr, tpr, lw=1.8, color="black", label=f"Proposed ({auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], ls=":", lw=0.7, color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(fontsize=6, loc="lower right"); plt.tight_layout()
    plt.savefig(config.REPORTS / "fig2_roc_curves.png"); plt.close()

    # ---------------------------------------------------------------- Fig 3, confusion
    cm = confusion_matrix(y, (final >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Benign", "Malicious"])
    ax.set_yticks([0, 1], ["Benign", "Malicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(config.REPORTS / "fig3_confusion_matrix.png"); plt.close()

    # ---------------------------------------------------------------- latency
    sample = urls[:200]
    pred.base_probabilities(sample[:8])                  # warm up
    t0 = time.perf_counter()
    for u in sample:
        pred.score([u])
    full_ms = (time.perf_counter() - t0) / len(sample) * 1000

    from . import bert_branch
    t0 = time.perf_counter()
    for u in sample:
        bert_branch.infer(pred.bert, pred.tok, [u], max_len=pred.cfg.get("max_len", 64))
    bert_ms = (time.perf_counter() - t0) / len(sample) * 1000

    # ---------------------------------------------------------------- adversarial
    mal = [u for u, t in zip(urls, y) if t == 1][:2000]
    rows = []
    for name, fn in [("None (baseline)", None)] + list(adversarial.ATTACKS.items()):
        pert = mal if fn is None else adversarial.perturb(mal, name)
        p = batched_probs(pred, pert)
        row = {"Attack": name}
        for k in config.BASE_ORDER:
            row[PRETTY[k]] = float((p[k] >= 0.5).mean())
        row["Proposed"] = float((ensemble.apply_meta(pred.meta, p) >= 0.5).mean())
        rows.append(row)
        print(f"  {name}: proposed recall {row['Proposed']:.3f}")
    adv = pd.DataFrame(rows).set_index("Attack")
    adv.to_csv(config.REPORTS / "adversarial_robustness.csv")

    plt.figure(figsize=(3.4, 2.6))
    x = np.arange(len(adv))
    for k in list(PRETTY.values()) + ["Proposed"]:
        plt.plot(x, adv[k], marker="o", ms=3, lw=1.6 if k == "Proposed" else 0.9,
                 color="black" if k == "Proposed" else None,
                 alpha=1 if k == "Proposed" else 0.7, label=k)
    plt.xticks(x, [i.split()[0] for i in adv.index], rotation=30, ha="right", fontsize=7)
    plt.ylabel("Recall on malicious URLs"); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(config.REPORTS / "fig4_adversarial.png"); plt.close()

    # ---------------------------------------------------------------- tables
    comp = pd.read_csv(config.REPORTS / "model_comparison.csv")
    comp["model"] = comp["model"].replace(PRETTY)
    lines = ["TABLE III  (paste into the paper, values rounded to 4 dp)", ""]
    lines.append(f"{'Model':<32}{'Acc.':>8}{'Prec.':>8}{'Rec.':>8}{'F1':>8}{'AUC':>8}")
    for _, r in comp.iterrows():
        lines.append(f"{r['model']:<32}{r['accuracy']:>8.4f}{r['precision']:>8.4f}"
                     f"{r['recall']:>8.4f}{r['f1']:>8.4f}{r['auc']:>8.4f}")
    lines += ["", "TABLE IV  recall under evasion attacks", "",
              adv.round(4).to_string(), ""]
    (config.REPORTS / "paper_tables.txt").write_text("\n".join(lines))

    best = comp.iloc[comp["f1"].idxmax()]
    prop = comp[comp["model"].str.contains("confidence", case=False)].iloc[0]
    numbers = {
        "test_urls": len(urls),
        "test_malicious": int(y.sum()),
        "proposed_accuracy_pct": round(float(prop["accuracy"]) * 100, 2),
        "proposed_f1": round(float(prop["f1"]), 4),
        "proposed_auc": round(float(prop["auc"]), 4),
        "best_single_model": str(best["model"]),
        "latency_full_ms": round(full_ms, 1),
        "latency_bert_ms": round(bert_ms, 1),
        "confusion_matrix": cm.tolist(),
        "training_rows": pred.cfg.get("n_rows"),
        "n_features": pred.cfg.get("n_features"),
    }
    (config.REPORTS / "paper_numbers.txt").write_text(json.dumps(numbers, indent=2))
    print("\n", json.dumps(numbers, indent=2))
    print("\nper-class report\n", classification_report(
        y, (final >= 0.5).astype(int), target_names=["benign", "malicious"], digits=4))
    print("figures and tables written to", config.REPORTS)


if __name__ == "__main__":
    main()
