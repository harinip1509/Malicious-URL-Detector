"""Every path, constant and hyperparameter lives here.

Training and serving both import this module, so there is exactly one place
where the model name, sequence length and base-model order are defined. That
matters more than it looks: if serving disagrees with training about any of
them, the API still returns confident answers, they are just wrong.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

BERT_NAME = "distilbert-base-uncased"
MAX_LEN = 64

# Order is fixed: the meta-learner's coefficients are positional.
BASE_ORDER = ["bert", "xgboost", "lightgbm", "catboost", "adaboost"]

BENIGN_LABELS = {"benign", "good", "legitimate", "safe", "0"}

ARTIFACT_FILES = {
    "bert": "bert",
    "xgboost": "xgboost.json",
    "lightgbm": "lightgbm.txt",
    "catboost": "catboost.cbm",
    "adaboost": "adaboost.joblib",
    "meta": "meta.joblib",
    "scaler": "scaler.joblib",
    "config": "config.json",
}


@dataclass
class TrainConfig:
    csv: str = ""
    max_rows: int = 60_000
    seed: int = 42

    # BERT branch
    epochs: int = 2
    batch_size: int = 64
    lr: float = 3e-5
    max_len: int = MAX_LEN

    # boosting branch
    n_estimators: int = 500
    max_depth: int = 8
    boost_lr: float = 0.08
    early_stopping: int = 30

    # splits
    test_size: float = 0.15
    val_size: float = 0.15

    skip_bert: bool = False          # reuse an already fine-tuned BERT
    run_adversarial: bool = True
    tags: list = field(default_factory=list)


def ensure_dirs():
    for d in (ARTIFACTS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
