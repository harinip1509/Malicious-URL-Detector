# Malicious URL Detection with a Hybrid BERT and Boosting Ensemble

A fine-tuned DistilBERT branch and four gradient boosting models combined by a
confidence-weighted stacking meta-learner, with a Flask API and a React front
end for the demo.

```
malicious-url-detector/
├── src/
│   ├── config.py         paths, model names, hyperparameters, base-model order
│   ├── features.py       34 lexical and host features, shared by train and serve
│   ├── data.py           loading, leakage removal, stratified splitting
│   ├── bert_branch.py    fine-tuning and CLS embedding extraction
│   ├── ensemble.py       boosters, confidence weighting, meta-learner, metrics
│   ├── adversarial.py    five evasion attacks for the robustness test
│   ├── predictor.py      inference class used by both the API and evaluation
│   ├── train.py          the whole pipeline, end to end
│   └── eda.py            dataset figures for the report
├── backend/app.py        Flask API
├── frontend/             Vite and React single page
├── artifacts/            trained models land here (gitignored)
├── reports/              CSV tables and figures for the paper
├── data/                 put malicious_phish.csv here
├── requirements.txt
└── Makefile
```

## Quick start on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/ and tick
   **"Add python.exe to PATH"** on the first installer screen.
2. Right-click `malicious-url-detector.zip`, choose **Extract All**, and open the
   extracted `malicious-url-detector` folder.
3. Double-click **`run.bat`**.

The first run takes a minute while it installs packages; after that your browser
opens http://localhost:5050. Keep the black console window open while you use the
app, and close it (or press Ctrl+C) to stop. If Windows SmartScreen warns about
the file, choose **More info → Run anyway**. For a different port, open a terminal
in the folder and run `set PORT=5051 && run.bat`. Once `artifacts\` holds trained
models, run `run.bat trained`.

## Quick start on a Mac

```bash
unzip malicious-url-detector.zip && cd malicious-url-detector
./run.sh
```

That creates a virtual environment, installs the few packages the demo needs,
starts the server and opens http://localhost:5050 in your browser. Nothing else
to install: the built interface ships in `frontend/dist/` and Flask serves it.

If macOS says the script cannot be opened, run `chmod +x run.sh` once. If port
5050 is taken, use `PORT=5051 ./run.sh`. The app uses 5050 rather than 5000
because macOS AirPlay Receiver occupies port 5000.

Once `artifacts/` holds trained models, start it with `./run.sh trained`.

## Setup

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Download `malicious_phish.csv` from the Kaggle Malicious URLs dataset into
`data/`. Any CSV with a URL column and a label column works; the loader finds
them by name.

## Demo without training

To show the working app before the models are trained:

```bash
pip install flask flask-cors numpy
DEMO=1 python3 -m backend.app         # open http://localhost:5050
```

On Windows PowerShell use `$env:DEMO="1"; python3 -m backend.app`.

The built UI ships in `frontend/dist/` and Flask serves it, so this is one server
and one URL with no Node step. In demo mode each "model" is a fixed rule over the
real lexical features and the final score passes through the real
confidence-weighting formula, but nothing is learned, and the UI labels this
clearly. Never quote demo numbers in the report. Once `artifacts/` holds trained
models, start the API without `DEMO=1` and it loads them automatically; if
loading fails it falls back to demo mode and says why.

If you edit the React code, rebuild with `cd frontend && npm install && npm run build`.

## Run it

```bash
make eda      # dataset figures, writes reports/
make train    # the full pipeline, writes artifacts/ and reports/
make api      # http://localhost:5050
make ui       # http://localhost:5173
```

Or without make:

```bash
python3 -m src.eda   --csv data/malicious_phish.csv
python3 -m src.train --csv data/malicious_phish.csv --max-rows 60000
python3 -m backend.app
cd frontend && npm run dev
```

Train on Colab or Kaggle with a GPU. Start at `--max-rows 60000` to confirm the
pipeline runs end to end, then raise it; the full 650k rows take a few hours on
a T4. Use `--skip-bert` to re-run the boosting and stacking stages against a
BERT you already fine-tuned.

## What training produces

`reports/model_comparison.csv` holds the baseline table: each of the five base
models alone, majority voting, static stacking, and the confidence-weighted
stack. `reports/adversarial_robustness.csv` holds recall on malicious URLs after
each evasion attack. Those two tables are the results section of the paper.

## The confidence weighting

This is the contribution, so it is worth stating precisely. A plain stack hands
the meta-learner five raw probabilities and learns one fixed weight per model.
But a model outputting 0.51 is guessing and one outputting 0.98 is not, and a
fixed weight cannot distinguish them. Each base probability is therefore pulled
toward neutral in proportion to how unsure the model is:

```
c_i = 2 * |p_i - 0.5|              confidence, in [0, 1]
z_i = c_i * (p_i - 0.5) + 0.5      damped probability
```

The meta-learner receives `[z_1..z_5, c_1..c_5]`. An uncertain vote decays toward
0.5 rather than counting in full, and the confidences are themselves features, so
the meta-learner can learn which models are worth listening to when they
hesitate. The implementation lives in `ensemble.confidence_weight` and is called
from both training and serving, so the two cannot drift apart.

## Domain reputation

Lexical features alone cannot tell a long, legitimate Google sign-in URL from a
phishing page: both are long, full of encoded characters, and contain words like
"signin". `src/reputation.py` supplies the missing fact, who owns the host, in
two directions. A host on a trusted registered domain (youtube.com,
accounts.google.com) is not flagged; a host that uses a known brand name, including
digit swaps like g00gle or paypa1, on a domain that brand does not own is pushed
towards malicious. Matching is on the registered domain, never a substring, so
`youtube.com.free-premium-gift.xyz` is caught rather than trusted.

With trained models loaded this runs as a policy after the ensemble: a trusted
domain caps the final score at 0.35 and the response sets
`reputation_override: true`, while each model's own vote is left untouched in the
breakdown. The known limitation is that trusting a domain also trusts its open
redirects, since the prototype never fetches a page.

## Decisions worth defending in a viva

**The meta-learner is fitted on validation probabilities, never training ones.**
The boosters have already memorised the training rows and output near 0 or 1
there, which teaches the meta-learner nothing about how much to trust each model
on data it has not seen.

**URLs labelled both benign and malicious are dropped before deduplication.**
Deduplicating first would delete the second copy and hide the conflict, leaving
one arbitrarily-labelled row to land in a split. This kind of leakage inflates
every reported score.

**Feature extraction lives in one module imported by both sides.** If training
and serving ever computed features differently, the API would return confident
answers that are quietly wrong, with no error raised anywhere.

**The task is binary.** Benign against everything else. For the four-class
version, change the mapping in `data.load`, set `num_labels=4` in
`bert_branch.fine_tune`, and replace `2*|p-0.5|` with a confidence based on the
maximum softmax probability, since the current formula only means anything for
two classes.

## Limitations

The system reads the URL string and never fetches the page, so a malicious site
behind a clean-looking link will pass. The adversarial results show how far each
model falls under simple string rewriting, which is the honest way to report
this rather than quoting only clean-test accuracy.
