"""Inference. One class, loaded once, used by both the API and the evaluation
scripts so that what you measure is exactly what you serve.
"""

import json
import time

import joblib
import numpy as np

from . import bert_branch, config, ensemble, features, reputation


REPUTATION_CAP = 0.35   # final score ceiling for trusted registered domains


class Predictor:
    def __init__(self, artifacts=None):
        art = config.ARTIFACTS if artifacts is None else artifacts
        self.art = art

        with open(art / config.ARTIFACT_FILES["config"]) as f:
            self.cfg = json.load(f)

        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier

        self.bert, self.tok = bert_branch.load(art / config.ARTIFACT_FILES["bert"])

        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(art / config.ARTIFACT_FILES["xgboost"]))
        self.lgb = lgb.Booster(model_file=str(art / config.ARTIFACT_FILES["lightgbm"]))
        self.cat = CatBoostClassifier().load_model(str(art / config.ARTIFACT_FILES["catboost"]))
        self.ada = joblib.load(art / config.ARTIFACT_FILES["adaboost"])
        self.meta = joblib.load(art / config.ARTIFACT_FILES["meta"])
        self.scaler = joblib.load(art / config.ARTIFACT_FILES["scaler"])

    # ---------------------------------------------------------------- core
    def base_probabilities(self, urls):
        """Every base model's probability of malicious, as a dict of arrays."""
        p_bert, emb = bert_branch.infer(self.bert, self.tok, urls,
                                        max_len=self.cfg.get("max_len", config.MAX_LEN))
        lex = self.scaler.transform(features.extract_batch(urls))
        X = np.hstack([lex, emb]).astype(np.float32)
        return {
            "bert": p_bert,
            "xgboost": self.xgb.predict_proba(X)[:, 1],
            "lightgbm": np.asarray(self.lgb.predict(X)),
            "catboost": self.cat.predict_proba(X)[:, 1],
            "adaboost": self.ada.predict_proba(X)[:, 1],
        }

    def score(self, urls):
        """Final ensemble probability for each URL."""
        return ensemble.apply_meta(self.meta, self.base_probabilities(urls))

    # ---------------------------------------------------------------- API shape
    def explain(self, urls):
        t0 = time.perf_counter()
        probs = self.base_probabilities(urls)
        final = ensemble.apply_meta(self.meta, probs)
        P = ensemble.stack_matrix(probs)
        C = 2.0 * np.abs(P - 0.5)
        elapsed = (time.perf_counter() - t0) * 1000

        # Reputation policy, applied after the models and reported openly: a URL
        # whose registered domain is on the trusted list is not flagged, however
        # its lexical surface looks. The individual model votes are left as they
        # were, so the breakdown still shows what the models themselves thought.
        overridden = np.zeros(len(urls), dtype=bool)
        for i, url in enumerate(urls):
            if reputation.trusted_domain(url) and final[i] >= 0.5:
                final[i] = min(final[i], REPUTATION_CAP)
                overridden[i] = True

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
                "reputation_override": bool(overridden[i]),
            })
        return results, round(elapsed, 1)
