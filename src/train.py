"""Train the whole system end to end.

    python -m src.train --csv data/malicious_phish.csv --max-rows 60000

Phases, in order: load and clean, split, fine-tune BERT, build the joint feature
matrix, train the four boosters, fit the meta-learner on validation
probabilities, evaluate against every baseline, run the adversarial test, save
artifacts. Results land in reports/ as CSV, ready for the paper.
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from . import adversarial, bert_branch, config, data, ensemble, features


def build_matrix(urls, bert, tok, scaler=None, max_len=config.MAX_LEN):
    """Lexical features and the BERT CLS embedding, side by side."""
    lex = features.extract_batch(urls)
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(lex)
    lex = scaler.transform(lex)
    p_bert, emb = bert_branch.infer(bert, tok, list(urls), max_len=max_len)
    return np.hstack([lex, emb]).astype(np.float32), p_bert, scaler


def run(cfg: config.TrainConfig):
    config.ensure_dirs()
    np.random.seed(cfg.seed)
    started = time.time()
    print("device:", bert_branch.DEVICE)

    print("\n[1/7] loading data")
    df = data.load(cfg.csv, cfg.max_rows, cfg.seed)
    train_df, val_df, test_df = data.split(df, cfg)

    print("\n[2/7] BERT branch")
    bert_dir = config.ARTIFACTS / config.ARTIFACT_FILES["bert"]
    if cfg.skip_bert and bert_dir.exists():
        print("  reusing the fine-tuned model already in artifacts/")
        bert, tok = bert_branch.load(bert_dir)
    else:
        bert, tok = bert_branch.fine_tune(train_df, val_df, cfg, bert_dir)

    print("\n[3/7] feature matrices")
    X_tr, p_tr, scaler = build_matrix(train_df["url"], bert, tok, max_len=cfg.max_len)
    X_va, p_va, _ = build_matrix(val_df["url"], bert, tok, scaler, cfg.max_len)
    X_te, p_te, _ = build_matrix(test_df["url"], bert, tok, scaler, cfg.max_len)
    print(f"  {X_tr.shape[1]} columns "
          f"({len(features.FEATURE_NAMES)} lexical + {X_tr.shape[1] - len(features.FEATURE_NAMES)} embedding)")

    print("\n[4/7] boosting branch")
    boosters = ensemble.train_boosters(X_tr, train_df["y"].values,
                                       X_va, val_df["y"].values, cfg)

    prob_val = {"bert": p_va}
    prob_test = {"bert": p_te}
    for name, model in boosters.items():
        prob_val[name] = model.predict_proba(X_va)[:, 1]
        prob_test[name] = model.predict_proba(X_te)[:, 1]

    print("\n[5/7] meta-learner")
    meta = ensemble.fit_meta(prob_val, val_df["y"].values, weighted=True)
    flat = ensemble.fit_meta(prob_val, val_df["y"].values, weighted=False)
    p_final = ensemble.apply_meta(meta, prob_test, weighted=True)

    print("\n[6/7] test set results")
    y_te = test_df["y"].values
    rows = [ensemble.score(name, y_te, prob_test[name]) for name in config.BASE_ORDER]
    rows.append(ensemble.score("majority voting", y_te, ensemble.majority_vote(prob_test)))
    rows.append(ensemble.score("static stacking", y_te,
                               ensemble.apply_meta(flat, prob_test, weighted=False)))
    rows.append(ensemble.score("confidence-weighted stack", y_te, p_final))

    print("\n", classification_report(y_te, (p_final >= 0.5).astype(int),
                                      target_names=["benign", "malicious"], digits=4))
    print("confusion matrix (rows true, cols predicted)")
    print(confusion_matrix(y_te, (p_final >= 0.5).astype(int)))

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORTS / "model_comparison.csv", index=False)

    print("\n[7/7] saving artifacts")
    boosters["xgboost"].save_model(str(config.ARTIFACTS / config.ARTIFACT_FILES["xgboost"]))
    boosters["lightgbm"].booster_.save_model(str(config.ARTIFACTS / config.ARTIFACT_FILES["lightgbm"]))
    boosters["catboost"].save_model(str(config.ARTIFACTS / config.ARTIFACT_FILES["catboost"]))
    joblib.dump(boosters["adaboost"], config.ARTIFACTS / config.ARTIFACT_FILES["adaboost"])
    joblib.dump(meta, config.ARTIFACTS / config.ARTIFACT_FILES["meta"])
    joblib.dump(scaler, config.ARTIFACTS / config.ARTIFACT_FILES["scaler"])
    with open(config.ARTIFACTS / config.ARTIFACT_FILES["config"], "w") as f:
        json.dump({
            "bert_name": config.BERT_NAME,
            "max_len": cfg.max_len,
            "base_order": config.BASE_ORDER,
            "feature_names": features.FEATURE_NAMES,
            "n_rows": int(len(df)),
            "n_features": int(X_tr.shape[1]),
            "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        }, f, indent=2)

    # The test set is kept for the adversarial script so the same rows are reused.
    test_df.to_csv(config.REPORTS / "test_split.csv", index=False)

    if cfg.run_adversarial:
        print("\n[extra] adversarial robustness")
        run_adversarial(test_df, bert, tok, boosters, meta, scaler, cfg)

    print(f"\ndone in {(time.time() - started)/60:.1f} min")
    print("artifacts ->", config.ARTIFACTS)
    print("reports   ->", config.REPORTS)
    return results


def run_adversarial(test_df, bert, tok, boosters, meta, scaler, cfg, sample=2000):
    """Re-score perturbed malicious URLs and record how far each model falls."""
    mal = test_df[test_df["y"] == 1]
    mal = mal.sample(min(sample, len(mal)), random_state=cfg.seed)
    urls = mal["url"].tolist()
    y = np.ones(len(urls), dtype=int)

    def recall_of_all(url_list):
        X, p_bert, _ = build_matrix(url_list, bert, tok, scaler, cfg.max_len)
        probs = {"bert": p_bert}
        for name, m in boosters.items():
            probs[name] = m.predict_proba(X)[:, 1]
        out = {name: float((probs[name] >= 0.5).mean()) for name in config.BASE_ORDER}
        out["confidence-weighted stack"] = float(
            (ensemble.apply_meta(meta, probs) >= 0.5).mean())
        return out

    rows = [dict(attack="none (baseline)", **recall_of_all(urls))]
    for attack in adversarial.ATTACKS:
        print(f"  {attack}")
        rows.append(dict(attack=attack,
                         **recall_of_all(adversarial.perturb(urls, attack, cfg.seed))))

    df = pd.DataFrame(rows).set_index("attack")
    df.to_csv(config.REPORTS / "adversarial_robustness.csv")
    print("\nrecall on malicious URLs after each evasion "
          f"(n={len(urls)}, higher is better)")
    print(df.round(4).to_string())
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Train the hybrid URL detection ensemble")
    p.add_argument("--csv", required=True, help="dataset CSV with url and label columns")
    p.add_argument("--max-rows", type=int, default=60000, help="0 uses everything")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=config.MAX_LEN)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-bert", action="store_true",
                   help="reuse the fine-tuned BERT already in artifacts/")
    p.add_argument("--no-adversarial", action="store_true")
    a = p.parse_args()
    return config.TrainConfig(
        csv=a.csv, max_rows=a.max_rows, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, max_len=a.max_len, seed=a.seed, skip_bert=a.skip_bert,
        run_adversarial=not a.no_adversarial)


if __name__ == "__main__":
    run(parse_args())
