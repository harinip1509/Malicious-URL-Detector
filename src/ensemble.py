"""The boosting branch and the confidence-weighted stacking meta-learner.

Confidence weighting is the contribution of this project, so it is worth stating
precisely. A plain stack hands the meta-learner five raw probabilities and learns
one fixed weight per model. But a model outputting 0.51 is guessing and one
outputting 0.98 is not, and a fixed weight cannot distinguish them. Each base
probability is therefore pulled toward neutral in proportion to how unsure the
model is:

    c_i = 2 * |p_i - 0.5|              confidence, in [0, 1]
    z_i = c_i * (p_i - 0.5) + 0.5      damped probability

The meta-learner receives [z_1..z_5, c_1..c_5]. An uncertain vote decays toward
0.5 instead of counting in full, and the confidences are themselves features, so
the meta-learner can learn which models are worth listening to when they
hesitate.
"""

import numpy as np

from . import config

# scikit-learn is imported inside the functions that need it. Demo mode only
# uses confidence_weight and stack_matrix, so keeping the import lazy means the
# demo starts instantly and does not depend on scikit-learn at all.


# ------------------------------------------------------------ boosting branch
def train_boosters(X_train, y_train, X_val, y_val, cfg):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    models = {}

    print("  xgboost")
    models["xgboost"] = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
        learning_rate=cfg.boost_lr, subsample=0.9, colsample_bytree=0.8,
        tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=cfg.early_stopping,
        random_state=cfg.seed, n_jobs=-1)
    models["xgboost"].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    print("  lightgbm")
    models["lightgbm"] = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators + 100, num_leaves=63,
        learning_rate=cfg.boost_lr, subsample=0.9, colsample_bytree=0.8,
        random_state=cfg.seed, n_jobs=-1, verbose=-1)
    models["lightgbm"].fit(X_train, y_train, eval_set=[(X_val, y_val)],
                           callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])

    print("  catboost")
    models["catboost"] = CatBoostClassifier(
        iterations=cfg.n_estimators + 100, depth=cfg.max_depth,
        learning_rate=cfg.boost_lr, random_seed=cfg.seed, verbose=0,
        early_stopping_rounds=cfg.early_stopping)
    models["catboost"].fit(X_train, y_train, eval_set=(X_val, y_val))

    print("  adaboost")
    models["adaboost"] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=4),
        n_estimators=250, learning_rate=0.5, random_state=cfg.seed)
    models["adaboost"].fit(X_train, y_train)

    return models


# ------------------------------------------------------------ stacking
def confidence_weight(prob_matrix):
    """Turn an (n, 5) probability matrix into (n, 10) meta features."""
    P = np.asarray(prob_matrix, dtype=np.float64)
    C = 2.0 * np.abs(P - 0.5)
    Z = C * (P - 0.5) + 0.5
    return np.hstack([Z, C]).astype(np.float32)


def stack_matrix(prob_dict):
    return np.column_stack([prob_dict[k] for k in config.BASE_ORDER])


def fit_meta(prob_dict_val, y_val, weighted=True):
    """Fit the meta-learner on validation probabilities.

    Never on training probabilities: the boosters have already memorised those
    rows and their probabilities there sit at 0 or 1, which teaches the
    meta-learner nothing about how much to trust each model in the wild.
    """
    from sklearn.linear_model import LogisticRegression
    P = stack_matrix(prob_dict_val)
    X = confidence_weight(P) if weighted else P
    return LogisticRegression(max_iter=2000, C=1.0).fit(X, y_val)


def apply_meta(meta, prob_dict, weighted=True):
    P = stack_matrix(prob_dict)
    X = confidence_weight(P) if weighted else P
    return meta.predict_proba(X)[:, 1]


def majority_vote(prob_dict):
    return np.mean([(prob_dict[k] >= 0.5).astype(float)
                    for k in config.BASE_ORDER], axis=0)


# ------------------------------------------------------------ metrics
def score(name, y_true, y_prob, threshold=0.5, quiet=False):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }
    if not quiet:
        print(f"  {name:<28} acc {row['accuracy']:.4f}  P {row['precision']:.4f}  "
              f"R {row['recall']:.4f}  F1 {row['f1']:.4f}  AUC {row['auc']:.4f}")
    return row
