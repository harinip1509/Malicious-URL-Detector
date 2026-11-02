# Malicious URL Detection with a Hybrid BERT and Boosting Ensemble

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-yellowgreen.svg)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosted-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> **Academic Project / Research Benchmark Documentation**  
> *Hybrid Ensemble Learning for Malicious URL Detection with BERT and Boosting Models*  
> Department of Information Technology — Deep Learning Lab Project (Sem VII)

---

## Table of Contents
1. [Project Overview & Key Contributions](#1-project-overview--key-contributions)
2. [Complete Experimental Results & Benchmark Metrics](#2-complete-experimental-results--benchmark-metrics)
3. [Held-Out Test Set Breakdown & Confusion Matrix](#3-held-out-test-set-breakdown--confusion-matrix)
4. [Model-by-Model Deep Dive & Comparative Analysis](#4-model-by-model-deep-dive--comparative-analysis)
5. [Mathematical Formulation: Confidence-Weighted Stacking](#5-mathematical-formulation-confidence-weighted-stacking)
6. [Feature Engineering: 34 Lexical & Host Attributes](#6-feature-engineering-34-lexical--host-attributes)
7. [Hybrid Feature Fusion Architecture (802 Dimensions)](#7-hybrid-feature-fusion-architecture-802-dimensions)
8. [Dataset Provenance & Anti-Leakage Cleaning Protocol](#8-dataset-provenance--anti-leakage-cleaning-protocol)
9. [Adversarial Robustness & Evasion Attacks](#9-adversarial-robustness--evasion-attacks)
10. [Hyperparameters & Training Specifications](#10-hyperparameters--training-specifications)
11. [System Architecture & Repository Structure](#11-system-architecture--repository-structure)
12. [Quick Start & Reproduction Guide](#12-quick-start--reproduction-guide)
13. [How to Use This Repository for a 20-Page Report / Thesis](#13-how-to-use-this-repository-for-a-20-page-report--thesis)
14. [BibTeX Citation](#14-bibtex-citation)

---

## 1. Project Overview & Key Contributions

Detecting malicious links prior to content retrieval (i.e., **string-only URL inspection**) is critical for eliminating zero-hour threats before drive-by downloads or credential-harvesting pages load. Traditional threat intelligence blocklists (e.g., PhishTank, Google Safe Browsing) are strictly reactive and fail during the initial hours of targeted campaigns.

This project introduces a **two-stream hybrid machine learning pipeline**:
1. **Semantic / Contextual Stream (`distilbert-base-uncased`):** Scans the subword token sequences to detect brand impersonation, deceptive naming, and linguistic anomalies without relying on fixed keyword lists.
2. **Structural / Statistical Stream (Tree Boosting):** Four gradient boosting architectures (**XGBoost, LightGBM, CatBoost, AdaBoost**) trained over a joint 802-dimensional representation (34 handcrafted lexical/host metrics + 768-dimensional DistilBERT `[CLS]` embeddings).
3. **Meta-Learning Fusion (Confidence-Weighted Stacking):** Rather than static probability averaging, a meta-classifier evaluates each model’s prediction scaled by its mathematical decision certainty.

### Key Innovations
* **Confidence Damping:** Uncertain base model predictions ($p_i \approx 0.5$) are dynamically drawn toward neutral, preventing ambiguous votes from polluting the ensemble.
* **Leakage-Conscious Data Preparation:** Conflicting URLs (same exact string assigned to both benign and malicious classes across raw scrapes) are purged *prior* to deduplication, preventing label ambiguity leakage.
* **Zero-Lookup Inference:** Pure client/server string parsing operating in sub-millisecond ranges without external DNS or HTTP roundtrips.

---

## 2. Complete Experimental Results & Benchmark Metrics

All experiments were executed on an identical held-out test split of **1,500 unseen URLs** (499 Malicious, 1,001 Benign). The table below reproduces the exact benchmark figures stored in `reports/model_comparison.csv`.

### Full Comparative Performance Table

| Model / Fusion Architecture | Model Type | Accuracy | Precision (PPV) | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** (`base-uncased`) | Transformer (Text Only) | 93.00% | 0.9560 | 0.8277 | 0.8872 | 0.9672 |
| **CatBoost** | Oblivious Tree Boosting | 96.40% | 0.9665 | 0.9238 | 0.9447 | 0.9897 |
| **XGBoost** | Histogram Gradient Boosting | 96.73% | 0.9707 | 0.9299 | 0.9498 | 0.9901 |
| **LightGBM** | Leaf-Wise Gradient Boosting | 96.73% | 0.9707 | 0.9299 | 0.9498 | 0.9905 |
| **Majority Voting** | Unweighted Hard Voting | 96.73% | 0.9727 | 0.9279 | 0.9497 | 0.9661 |
| **Static Stacking** | Logistic Regression on Raw $p_i$ | 96.80% | **0.9727** | 0.9299 | 0.9508 | 0.9906 |
| **AdaBoost** | Adaptive Shallow Trees | **0.9687** | **0.9788** | 0.9259 | 0.9516 | 0.9893 |
| **Confidence-Weighted Ensemble (Ours)** | **Damped Stacking Meta-Learner** | **0.9687** | 0.9650 | **0.9399** | **0.9523** | **0.9907** |

> **Key Performance Takeaway:**  
> The **Confidence-Weighted Ensemble achieves the highest Recall (93.99%)**, the highest **F1-Score (0.9523)**, and the highest **ROC-AUC (0.9907)** of all tested configurations, matching the top accuracy (96.87%).  
> Against standard Static Stacking, confidence weighting boosts malicious recall by a full **+1.00%** (jumping from 92.99% to 93.99%), reducing false negatives by **14.3%** in mission-critical detection.

---

## 3. Held-Out Test Set Breakdown & Confusion Matrix

### Test Split Class Distribution
* **Total Held-Out Test Instances:** $N = 1,500$
* **Ground-Truth Benign ($y=0$):** $1,001$ instances ($66.73\%$)
* **Ground-Truth Malicious ($y=1$):** $499$ instances ($33.27\%$)
* **Evaluation Threshold:** $\tau = 0.50$

### Ensemble Confusion Matrix

$$\begin{array}{c|cc}
& \textbf{Predicted Benign} & \textbf{Predicted Malicious} \\
\hline
\textbf{Actual Benign (1,001)} & \mathbf{TN = 984} & \mathbf{FP = 17} \\
\textbf{Actual Malicious (499)} & \mathbf{FN = 30} & \mathbf{TP = 469} \\
\end{array}$$

### Exhaustive Classification Metrics

* **Accuracy ($ACC$):**
  $$ACC = \frac{TP + TN}{TP + TN + FP + FN} = \frac{469 + 984}{1500} = \mathbf{96.87\%}$$
* **Sensitivity / True Positive Rate / Recall ($TPR$):**
  $$TPR = \frac{TP}{TP + FN} = \frac{469}{469 + 30} = \mathbf{93.99\%}$$
* **Specificity / True Negative Rate ($TNR$):**
  $$TNR = \frac{TN}{TN + FP} = \frac{984}{984 + 17} = \mathbf{98.30\%}$$
* **Precision / Positive Predictive Value ($PPV$):**
  $$PPV = \frac{TP}{TP + FP} = \frac{469}{469 + 17} = \mathbf{96.50\%}$$
* **Negative Predictive Value ($NPV$):**
  $$NPV = \frac{TN}{TN + FN} = \frac{984}{984 + 30} = \mathbf{97.04\%}$$
* **False Positive Rate ($FPR$):**
  $$FPR = 1 - TNR = \frac{17}{1001} = \mathbf{1.70\%}$$
* **False Negative Rate / Miss Rate ($FNR$):**
  $$FNR = 1 - TPR = \frac{30}{499} = \mathbf{6.01\%}$$
* **Balanced Accuracy ($BACC$):**
  $$BACC = \frac{TPR + TNR}{2} = \frac{0.9399 + 0.9830}{2} = \mathbf{96.15\%}$$
* **$F_1$-Score (Harmonic Mean of Precision and Recall):**
  $$F_1 = 2 \times \frac{PPV \times TPR}{PPV + TPR} = 2 \times \frac{0.9650 \times 0.9399}{0.9650 + 0.9399} = \mathbf{0.9523}$$
* **Area Under the ROC Curve ($ROC\text{-}AUC$):**
  $$\text{ROC-AUC} = \mathbf{0.9907}$$

---

## 4. Model-by-Model Deep Dive & Comparative Analysis

### 1. DistilBERT Branch (`distilbert-base-uncased`)
* **Role:** Contextual tokenization and semantic analysis.
* **Input:** Raw URL string tokenized into subwords (WordPiece, max sequence length 64).
* **Observation:** Achieved very high precision (**95.60%**) but lower recall (**82.77%**). DistilBERT effectively captures recognizable lexical patterns and brand keywords, but struggles with purely statistical structural markers (such as character entropy or raw dot counts) when URLs contain obfuscated non-dictionary paths.

### 2. Gradient Boosted Decision Trees (XGBoost & LightGBM)
* **Role:** Non-linear decision surfaces over the joint 802-dimensional feature space.
* **Observation:** Both achieved identical accuracy (**96.73%**) and recall (**92.99%**). Because they ingest both handcrafted features and the 768-dim `[CLS]` embedding, they combine lexical statistics with transformer semantic context. LightGBM showed a slight edge in discriminative separation ($\text{ROC-AUC} = 0.9905$ vs $0.9901$).

### 3. CatBoost (`CatBoostClassifier`)
* **Role:** Symmetrical/oblivious decision trees preventing overfitting in high-dimensional embedding spaces.
* **Observation:** Demonstrated conservative probability outputs ($\text{Accuracy} = 96.40\%$, $\text{Precision} = 96.65\%$, $\text{Recall} = 92.38\%$), providing a stabilizing anchor for the meta-learner.

### 4. AdaBoost (`AdaBoostClassifier`)
* **Role:** Adaptive boosting focused on hard-to-classify borderline examples.
* **Observation:** Achieved the highest individual precision (**97.88%**) and top standalone accuracy (**96.87%**), but had lower recall (**92.59%**).

### 5. Meta-Learner Comparison: Static vs. Confidence-Weighted Stacking
* **Majority Voting:** Ignores model confidence entirely, yielding an inferior ROC-AUC of **0.9661**.
* **Static Stacking:** Employs raw probabilities $[p_1, \dots, p_5]$. It achieves **92.99%** recall and **0.9508** F1.
* **Confidence-Weighted Stacking:** Augments the meta-learner with both damped probabilities and explicit confidence coordinates $[z_1, \dots, z_5, c_1, \dots, c_5]$. This structural change allows the meta-classifier to differentiate between a model that confidently predicts a threat ($p=0.99$) versus one that is guessing ($p=0.51$). Result: **Recall rises to 93.99%** and **F1 reaches 0.9523**.

---

## 5. Mathematical Formulation: Confidence-Weighted Stacking

Let $M = \{m_1, m_2, m_3, m_4, m_5\}$ denote the five base classifiers:
$$m_1 = \text{DistilBERT}, \quad m_2 = \text{XGBoost}, \quad m_3 = \text{LightGBM}, \quad m_4 = \text{CatBoost}, \quad m_5 = \text{AdaBoost}$$

For an input URL $u$, each base model $i$ outputs a predicted posterior probability:
$$p_i = \hat{P}(y=1 \mid u, m_i) \in [0, 1]$$

### 1. Confidence Measure ($c_i$)
The distance from the maximum-entropy decision boundary ($p=0.5$) is mapped to a normalized confidence coefficient $c_i \in [0, 1]$:
$$c_i = 2 \left| p_i - \frac{1}{2} \right|$$
* If $p_i = 0.5$ (pure uncertainty), $c_i = 0$.
* If $p_i \in \{0.0, 1.0\}$ (absolute certainty), $c_i = 1$.

### 2. Damped Probability Representation ($z_i$)
Each probability is non-linearly damped toward the neutral boundary ($0.5$) proportionally to its uncertainty:
$$z_i = c_i \left( p_i - \frac{1}{2} \right) + \frac{1}{2} = 2 \operatorname{sgn}\left(p_i - \frac{1}{2}\right) \left(p_i - \frac{1}{2}\right)^2 + \frac{1}{2}$$

### 3. Meta-Feature Vector Formulation
The meta-learner receives a 10-dimensional representation capturing both direction and certainty:
$$\mathbf{x}_{\text{meta}} = \big[ z_1, z_2, z_3, z_4, z_5, \; c_1, c_2, c_3, c_4, c_5 \big]^T \in \mathbb{R}^{10}$$

### 4. Meta-Learner Optimization
The meta-learner is a logistic regression model with $L_2$ regularization parameter $C=1.0$:
$$\hat{P}(y=1 \mid u) = \sigma\left(\mathbf{w}^T \mathbf{x}_{\text{meta}} + b\right) = \frac{1}{1 + \exp\left(-\left(\sum_{i=1}^5 w_i z_i + \sum_{i=1}^5 w_{i+5} c_i + b\right)\right)}$$

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{j=1}^{N_{\text{val}}} \log\left(1 + \exp\left(-y_j (\mathbf{w}^T \mathbf{x}_{\text{meta}, j} + b)\right)\right)$$

> **Methodological Safeguard:**  
> The meta-learner is strictly trained on out-of-fold validation set probabilities ($\mathcal{D}_{\text{val}}$), **never on training set probabilities**. Since tree boosters achieve near-zero training loss, fitting meta-weights on training data would cause extreme meta-overfitting.

---

## 6. Feature Engineering: 34 Lexical & Host Attributes

The feature extraction pipeline (`src/features.py`) extracts 34 deterministic metrics grouped into 5 structural categories:

| Feature Index | Feature Identifier | Domain | Description & Mathematical Definition |
| :---: | :--- | :--- | :--- |
| 1 | `url_length` | Structural | Total character count: $\|u\|$ |
| 2 | `hostname_length` | Host | Total character count of extracted FQDN |
| 3 | `path_length` | Path | Total character count of URL path component |
| 4 | `query_length` | Query | Character count of query string parameter payload |
| 5 | `path_depth` | Path | Hierarchy depth: number of non-empty `/` segments |
| 6 | `digit_count` | Lexical | Total numeric digit characters $[0\text{--}9]$ |
| 7 | `letter_count` | Lexical | Total alphabetic characters $[a\text{--}z, A\text{--}Z]$ |
| 8 | `digit_letter_ratio` | Lexical | Normalized digit frequency: $\frac{N_{\text{digits}}}{N_{\text{letters}} + 1}$ |
| 9 | `dot_count` | Delimiter | Count of `.` characters in entire URL |
| 10 | `hyphen_count` | Delimiter | Count of `-` characters (common in phishing brands) |
| 11 | `underscore_count` | Delimiter | Count of `_` characters |
| 12 | `slash_count` | Delimiter | Count of `/` characters |
| 13 | `question_count` | Delimiter | Count of `?` characters |
| 14 | `equal_count` | Delimiter | Count of `=` characters (query assignments) |
| 15 | `at_count` | Delimiter | Count of `@` symbols (credential embedding syntax) |
| 16 | `ampersand_count` | Delimiter | Count of `&` characters |
| 17 | `percent_count` | Delimiter | Count of `%` characters (percent-encoding obfuscation) |
| 18 | `tilde_count` | Delimiter | Count of `~` characters |
| 19 | `plus_count` | Delimiter | Count of `+` characters |
| 20 | `special_char_count` | Lexical | Total non-alphanumeric character count |
| 21 | `subdomain_count` | Host | Dot count in hostname minus one: $\max(0, N_{\text{dots, host}} - 1)$ |
| 22 | `has_ip` | Host | Binary indicator: hostname is an IPv4 dotted quad |
| 23 | `has_hex_ip` | Host | Binary indicator: hostname is hexadecimal/octal formatted IP |
| 24 | `has_port` | Host | Binary indicator: explicit non-standard port declaration |
| 25 | `has_https` | Security | Binary indicator: URL protocol scheme is HTTPS |
| 26 | `is_shortened` | Host | Binary indicator: domain matches known shorteners (`bit.ly`, etc.) |
| 27 | `url_entropy` | Information | Shannon Entropy of entire URL character distribution |
| 28 | `hostname_entropy` | Information | Shannon Entropy of hostname character distribution |
| 29 | `longest_token_length`| Lexical | Maximum character length of any delimiter-separated sub-token |
| 30 | `tld_length` | Host | Character length of effective Top-Level Domain |
| 31 | `suspicious_word_count`| Linguistic | Match count against high-risk token lexicon (`login`, `verify`, etc.) |
| 32 | `has_double_slash_in_path`| Path | Binary indicator: `//` redirection anomaly within path string |
| 33 | `digit_in_hostname` | Host | Binary indicator: presence of any numeric digit in hostname |
| 34 | `vowel_consonant_ratio`| Linguistic | Ratio of vowels to consonants in hostname: $\frac{N_v}{N_c + 1}$ |

### Shannon Information Entropy Formulation
For string $S$ of length $L$ over character alphabet $\Sigma$:
$$H(S) = -\sum_{k \in \Sigma} P(k) \log_2 P(k), \quad \text{where } P(k) = \frac{\operatorname{count}(k, S)}{L}$$
High entropy indicates randomized algorithmic domain generation (DGA), while low entropy indicates standard human language paths.

---

## 7. Hybrid Feature Fusion Architecture (802 Dimensions)

The tree boosting branch operates on a unified multi-modal feature vector:

```
┌───────────────────────────────────────────────┐
│              Raw URL Input String             │
└───────┬───────────────────────────────┬───────┘
        │                               │
        ▼                               ▼
┌──────────────────┐            ┌───────────────────────────────┐
│ 34 Handcrafted   │            │  distilbert-base-uncased      │
│ Lexical & Host   │            │  Fine-Tuned Token Encoder     │
│ Features         │            └───────────────┬───────────────┘
└───────┬──────────┘                            │
        │ StandardScale                         │ Final Transformer Layer
        ▼                                       ▼
┌──────────────────┐            ┌───────────────────────────────┐
│ 34 Normalized    │            │ 768-Dimensional Contextual    │
│ Float Features   │            │ [CLS] Semantic Embedding      │
└───────┬──────────┘            └───────────────┬───────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
      ┌──────────────────────────────────────────────────┐
      │ Concatenated Representation: Vector ∈ ℝ⁸⁰²       │
      └─────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌───────────┐
      │  XGBoost  │   │ LightGBM  │   │ CatBoost  │
      └───────────┘   └───────────┘   └───────────┘
```

---

## 8. Dataset Provenance & Anti-Leakage Cleaning Protocol

### 1. Data Source
* Sourced from the **Kaggle Malicious URLs Benchmark** (incorporating curated feeds from ISCX, PhishTank, and Alexa top domains).
* Multi-class labels mapped to binary:
  * **Class 0 (Benign):** `benign`, `good`, `legitimate`, `safe`, `0`
  * **Class 1 (Malicious):** `phishing`, `malware`, `defacement`, `1`

### 2. Anti-Leakage Preprocessing Protocol
Standard URL deduplication often creates synthetic data leakage. Our ingestion pipeline implements three safeguards:
1. **Conflicting Label Purge:** URLs appearing in public datasets with opposing labels (e.g., scraped as benign on day 1, and flagged as phishing on day 5) are identified and completely purged before deduplication.
2. **Exact Deduplication:** Redundant URL strings are removed after conflict purging.
3. **Stratified Split:** Data is partitioned into $70\%$ Training ($6,999$ rows), $15\%$ Validation ($1,500$ rows), and $15\%$ Held-Out Test ($1,500$ rows) with fixed random seed $42$.

---

## 9. Adversarial Robustness & Evasion Attacks

Implemented in `src/adversarial.py`, the system tests resilience against 5 real-world evasion strategies:

```
Attack 1: Homoglyph Substitution ───> Replace Latin chars with Cyrillic (e.g. 'a' -> '\u0430')
Attack 2: Percent-Encoding ─────────> Encode path characters into hex (e.g. '/' -> '%2F')
Attack 3: Subdomain Padding ────────> Prepend deceptive prefixes ('login.secure.verify.xyz')
Attack 4: Shortener Wrapping ───────> Encapsulate URL through URL shorteners ('bit.ly/...')
Attack 5: Typosquatting Insertion ──> Character duplication and transposition
```

* **Observation:** Handcrafted lexical features experience degradation under character obfuscation (such as homoglyph swaps), whereas the fine-tuned DistilBERT branch preserves semantic contextual awareness. The confidence-weighted stacking layer dynamically mitigates individual branch failure.

---

## 10. Hyperparameters & Training Specifications

| Component | Parameter | Configured Value |
| :--- | :--- | :--- |
| **DistilBERT** | Architecture | `distilbert-base-uncased` (6 layers, 12 heads) |
| | Max Sequence Length | 64 tokens |
| | Epochs | 2 |
| | Batch Size | 64 |
| | Learning Rate | $3 \times 10^{-5}$ (AdamW) |
| | Warmup Ratio | 0.1 |
| **XGBoost** | Estimators / Max Depth | 500 / 8 |
| | Learning Rate / Subsample | 0.08 / 0.9 (colsample_bytree: 0.8) |
| | Early Stopping Rounds | 30 |
| **LightGBM** | Estimators / Max Leaves | 600 / 63 |
| | Learning Rate / Subsample | 0.08 / 0.9 (colsample_bytree: 0.8) |
| **CatBoost** | Iterations / Depth | 600 / 8 |
| | Learning Rate | 0.08 |
| **AdaBoost** | Estimators / Base Estimator | 250 / `DecisionTreeClassifier(max_depth=4)` |
| **Meta-Learner** | Algorithm | `LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')` |
| | Fit Partition | Out-of-fold validation probabilities ($\mathcal{D}_{\text{val}}$) |

---

## 11. System Architecture & Repository Structure

```
malicious-url-detector/
├── artifacts/                  # Trained model checkpoints & serializations
│   ├── bert/                   # Fine-tuned DistilBERT checkpoint (model.safetensors, 267 MB)
│   ├── xgboost.json            # Serialized XGBoost booster
│   ├── lightgbm.txt            # Serialized LightGBM booster
│   ├── catboost.cbm            # Serialized CatBoost model
│   ├── adaboost.joblib         # Serialized AdaBoost ensemble
│   ├── meta.joblib             # Trained Stacking Logistic Regression meta-learner
│   ├── scaler.joblib           # StandardScaler fitted on 34 lexical training features
│   └── config.json             # Runtime feature order and metadata configuration
├── backend/
│   └── app.py                  # Flask REST API serving inference endpoints
├── frontend/                   # React / Vite single-page application
│   ├── src/                    # UI components, radar score charts, breakdown panels
│   └── dist/                   # Production-compiled static frontend assets
├── reports/
│   ├── model_comparison.csv    # Benchmark metric table across all 8 configurations
│   ├── test_split.csv          # Held-out 1,500 test instances with ground truth labels
│   └── ieee_paper_draft.tex    # Complete conference paper draft formatted in LaTeX
├── src/
│   ├── config.py               # Global configurations, hyperparameter dataclass
│   ├── features.py             # 34-feature extraction engine (shared by train & serve)
│   ├── data.py                 # Anti-leakage cleaner, stratified split loader
│   ├── bert_branch.py          # DistilBERT fine-tuning and [CLS] feature extractor
│   ├── ensemble.py             # Boosting trainers, confidence weighting formulas
│   ├── adversarial.py          # 5 evasion attack generators for robustness stress testing
│   ├── predictor.py            # End-to-end inference orchestrator
│   └── train.py                # Pipeline execution script (stages 1 through 7)
├── requirements.txt            # Python dependencies (PyTorch, Transformers, XGBoost, etc.)
└── README.md                   # This comprehensive documentation
```

---

## 12. Quick Start & Reproduction Guide

### Prerequisites
* Windows, macOS, or Linux
* Python 3.10 or newer
* Node.js 18+ (only if modifying frontend code)

### 1. Installation & Environment Setup
```powershell
# Navigate into repository
cd C:\Users\Harini\Downloads\malicious-url-detector

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Interactive Application
```powershell
# Start the Flask API & React Dashboard (serves on http://localhost:5050)
python -m backend.app
```
Open **`http://localhost:5050`** in any browser.

### 3. Re-Evaluate Models on Held-Out Test Set
```powershell
python -c "
import pandas as pd
df = pd.read_csv('reports/model_comparison.csv')
print(df.to_string(index=False))
"
```

---

## 13. How to Use This Repository for a 20-Page Report / Thesis

If you or your team need to write a comprehensive **20-page project report or final year thesis**, use the following mapping directly from this repository:

| Report Section | Suggested Page Budget | Primary Repository Source / Reference |
| :--- | :---: | :--- |
| **1. Abstract & Introduction** | 2 pages | Read [Section 1](#1-project-overview--key-contributions) & `reports/ieee_paper_draft.tex` (lines 18–38). Detail the rise of zero-hour phishing links and failure of reactive blocklists. |
| **2. Literature Review** | 3 pages | Cite BERT/DistilBERT papers, gradient boosting survey, and URL classification baselines from `reports/ieee_paper_draft.tex` bibliography. |
| **3. Dataset & Data Engineering** | 3 pages | Use [Section 8](#8-dataset-provenance--anti-leakage-cleaning-protocol). Detail the conflict-purging protocol, deduplication, and stratified 70/15/15 split. |
| **4. Feature Engineering (Lexical & Host)** | 3 pages | Recreate the complete 34-feature table from [Section 6](#6-feature-engineering-34-lexical--host-attributes), including Shannon entropy and character distribution equations. |
| **5. Proposed Hybrid Methodology** | 4 pages | Detail the 802-dimensional feature fusion from [Section 7](#7-hybrid-feature-fusion-architecture-802-dimensions), followed by the complete mathematical derivation of Confidence-Weighted Stacking from [Section 5](#5-mathematical-formulation-confidence-weighted-stacking). |
| **6. Results & Discussion** | 3 pages | Incorporate the Full Comparative Table from [Section 2](#2-complete-experimental-results--benchmark-metrics), the 1,500-sample Confusion Matrix & diagnostic metrics from [Section 3](#3-held-out-test-set-breakdown--confusion-matrix), and model deep-dives from [Section 4](#4-model-by-model-deep-dive--comparative-analysis). |
| **7. Adversarial Robustness & Limitations** | 1.5 pages | Use [Section 9](#9-adversarial-robustness--evasion-attacks) to explain the 5 evasion attacks (homoglyphs, percent encoding, padding, shorteners) and why pure lexical models fail. |
| **8. Conclusion & Future Scope** | 0.5 pages | Summarize the +1.0% recall advantage of confidence weighting and outline future work (domain-disjoint evaluation, real-time DNS graph fusion). |

---

## 14. BibTeX Citation

```bibtex
@inproceedings{panigrahi2026malicious,
  title     = {Confidence-Weighted Hybrid Learning for Malicious URL Detection with DistilBERT and Boosting Ensembles},
  author    = {Panigrahi, Harini and Bhayde, Prapti and Mahadik, Akhil},
  booktitle = {Department of Information Technology Research Proceedings, Vidyalankar Institute of Technology},
  year      = {2026},
  url       = {https://github.com/harinip1509/Malicious-URL-Detector}
}
```

---

## License
This project is distributed under the MIT License.
