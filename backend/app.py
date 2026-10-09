"""Flask API.

    python -m backend.app          # http://localhost:5050

Endpoints
    GET  /api/health   whether the models loaded, and what they were trained on
    POST /api/scan     {"url": "..."} or {"urls": [...]}, max 50 per request
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config          # noqa: E402
from src.demo import DemoPredictor    # noqa: E402

DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = Flask(__name__, static_folder=None)
CORS(app)

STATE = {"predictor": None, "error": None, "mode": None}


def boot():
    """Load once at startup. A failure is recorded rather than raised, so the
    UI can explain what is missing instead of showing a refused connection."""
    if os.environ.get("DEMO") == "1":
        STATE.update(predictor=DemoPredictor(), mode="demo")
        print("DEMO=1 set, running demo scorer")
        return
    try:
        from src.predictor import Predictor   # imports torch; kept lazy so demo mode runs without it
        STATE.update(predictor=Predictor(), mode="trained")
        print("models loaded from", config.ARTIFACTS)
    except Exception as e:                       # noqa: BLE001
        STATE["error"] = f"{type(e).__name__}: {e}"
        STATE.update(predictor=DemoPredictor(), mode="demo")
        print("could not load trained models, falling back to demo mode:", STATE["error"])


@app.get("/api/health")
def health():
    p = STATE["predictor"]
    if p is None:
        return jsonify(
            status="unavailable",
            error=STATE["error"],
            hint="Run python -m src.train --csv <dataset.csv>, then restart this server.",
        ), 503
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return jsonify(
        status="ready",
        mode=STATE["mode"],
        note=("Demo scorer: fixed rules over lexical features, not the trained ensemble."
              if STATE["mode"] == "demo" else None),
        device=device,
        models=config.BASE_ORDER,
        trained_at=p.cfg.get("trained_at"),
        training_rows=p.cfg.get("n_rows"),
        n_features=p.cfg.get("n_features"),
    )


@app.post("/api/scan")
def scan():
    p = STATE["predictor"]
    if p is None:
        return jsonify(error="Models are not loaded. " + (STATE["error"] or "")), 503

    body = request.get_json(silent=True) or {}
    urls = body.get("urls") or ([body["url"]] if body.get("url") else [])
    urls = [str(u).strip() for u in urls if str(u).strip()][:50]
    if not urls:
        return jsonify(error="Send a url, or a list of urls."), 400

    try:
        results, elapsed = p.explain(urls)
    except Exception as e:                       # noqa: BLE001
        return jsonify(error=f"Scan failed: {type(e).__name__}: {e}"), 500
    return jsonify(results=results, elapsed_ms=elapsed, mode=STATE["mode"])


@app.get("/")
@app.get("/<path:path>")
def ui(path=""):
    """Serve the built React app, so one command runs the whole demo."""
    if not DIST.exists():
        return ("Front end not built. Run: cd frontend && npm install && npm run build. "
                "Or use npm run dev on port 5173."), 404
    target = DIST / path
    if path and target.is_file():
        return send_from_directory(DIST, path)
    return send_from_directory(DIST, "index.html")


boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
