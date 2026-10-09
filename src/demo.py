"""Demo mode: lets the API and UI run before any model has been trained.

Each "model" here is a fixed logistic rule over the real lexical features, with a
different emphasis per model so the vote strip shows genuine disagreement. The
final score goes through the real confidence-weighting formula from ensemble.py.

This is a stand-in for showing the interface, not a result. The API reports
status "demo" and the UI labels every verdict accordingly. Never quote these
numbers in the report; train the real ensemble for that.
"""

import hashlib
import math
import time

import numpy as np

from . import config, ensemble, features, reputation

# Each rule weights the same features differently, loosely mirroring what each
# real branch tends to pick up on: BERT on words and structure, trees on counts.
RULES = {
    "bert": {"bias": -2.0, "suspicious_word_count": 1.1, "subdomain_count": 0.45,
             "hyphen_count": 0.25, "has_ip": 1.2, "is_shortened": 1.9, "at_count": 1.2},
    "xgboost": {"bias": -2.4, "has_ip": 2.0, "url_length": 0.022, "digit_in_hostname": 0.2,
                "has_https": -0.9, "has_port": 0.8, "suspicious_word_count": 0.6,
                "is_shortened": 1.1},
    "lightgbm": {"bias": -2.3, "url_entropy": 0.55, "dot_count": 0.25, "has_ip": 1.7,
                 "has_https": -0.8, "suspicious_word_count": 0.7, "is_shortened": 2.3},
    "catboost": {"bias": -2.2, "subdomain_count": 0.6, "tld_length": 0.15, "has_ip": 1.8,
                 "hyphen_count": 0.3, "suspicious_word_count": 0.8, "has_https": -0.7},
    "adaboost": {"bias": -1.6, "has_ip": 1.4, "url_length": 0.012, "has_https": -0.6,
                 "suspicious_word_count": 0.5, "special_char_count": 0.08},
}

# Fixed meta weights over [z_1..z_5, c_1..c_5], standing in for the fitted
# logistic regression. Positive weight on each damped probability, none on
# confidence alone, so the demo still shows the weighting doing its job.
META_W = np.array([2.2, 1.8, 1.8, 1.9, 1.3, 0, 0, 0, 0, 0], dtype=np.float64)
META_B = -4.5


def _jitter(url, name):
    """Small deterministic noise so identical rules do not give identical bars."""
    h = hashlib.md5(f"{name}:{url}".encode()).hexdigest()
    return (int(h[:6], 16) / 0xFFFFFF - 0.5) * 0.6


# Reputation weights per model. The semantic branch reads brand names in the
# text, so it reacts most strongly to impersonation; the tree models lean on it
# a little less.
TRUST_W = {"bert": -3.4, "xgboost": -3.0, "lightgbm": -3.0, "catboost": -3.2, "adaboost": -2.6}
IMPERSONATE_W = {"bert": 2.6, "xgboost": 1.6, "lightgbm": 1.8, "catboost": 2.0, "adaboost": 1.4}


def _prob(url, fv, name):
    rule = RULES[name]
    z = rule["bias"] + _jitter(url, name)
    for feat, w in rule.items():
        if feat != "bias":
            z += w * fv[feat]
    z += TRUST_W[name] * fv["trusted"] + IMPERSONATE_W[name] * fv["impersonates"]
    return 1.0 / (1.0 + math.exp(-z))


def _centred(url):
    """Raw features, with length and entropy measured from a typical benign URL
    so that a normal link does not accumulate risk just by existing."""
    fv = dict(zip(features.FEATURE_NAMES, features.extract(url)))
    fv["url_length"] = max(fv["url_length"] - 54.0, 0.0)
    fv["url_entropy"] = max(fv["url_entropy"] - 3.9, 0.0)
    fv["dot_count"] = max(fv["dot_count"] - 2.0, 0.0)
    fv["tld_length"] = max(fv["tld_length"] - 3.0, 0.0)
    fv["trusted"] = 1.0 if reputation.trusted_domain(url) else 0.0
    fv["impersonates"] = 1.0 if reputation.impersonated_brand(url) else 0.0
    return fv


class DemoPredictor:
    cfg = {"trained_at": None, "n_rows": None, "n_features": len(features.FEATURE_NAMES)}

    def base_probabilities(self, urls):
        out = {k: [] for k in config.BASE_ORDER}
        for u in urls:
            fv = _centred(u)
            for k in config.BASE_ORDER:
                out[k].append(_prob(u, fv, k))
        return {k: np.array(v) for k, v in out.items()}

    def explain(self, urls):
        t0 = time.perf_counter()
        probs = self.base_probabilities(urls)
        P = ensemble.stack_matrix(probs)
        X = ensemble.confidence_weight(P)
        final = 1.0 / (1.0 + np.exp(-(X @ META_W + META_B)))
        C = 2.0 * np.abs(P - 0.5)

        results = []
        for i, url in enumerate(urls):
            verdict = "malicious" if final[i] >= 0.5 else "benign"
            models = [{
                "model": name,
                "probability": round(float(P[i, j]), 4),
                "confidence": round(float(C[i, j]), 4),
                "vote": "malicious" if P[i, j] >= 0.5 else "benign",
            } for j, name in enumerate(config.BASE_ORDER)]
            results.append({
                "url": url,
                "score": round(float(final[i]), 4),
                "verdict": verdict,
                "confidence": round(float(abs(final[i] - 0.5) * 2), 4),
                "agreement": round(sum(m["vote"] == verdict for m in models) / len(models), 2),
                "models": models,
                "signals": reputation.reputation_signals(url) + features.notable_flags(url),
                "demo": True,
            })
        return results, round((time.perf_counter() - t0) * 1000, 1)
