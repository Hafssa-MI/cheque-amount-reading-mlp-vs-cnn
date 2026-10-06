# Automatic Cheque Amount Reading: MLP vs CNN

> **Case Study 1, Advanced Artificial Intelligence (2026-2027)

Reading the handwritten amount on a bank cheque position by position (units, tens, hundreds, thousands, ten-thousands) with a **Multilayer Perceptron (MLP)**, understanding *why it fails*, and testing whether a **Convolutional Neural Network (CNN)** fixes it. The study also asks when the model's reading can be trusted, which decides how many cheques can be processed automatically.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-yellow)
![Language](https://img.shields.io/badge/Notebook-French-lightgrey)

---

## Overview

The data comes from a fictional bank, **Banque Atlas**. Each sample is a 32 × 128 grayscale image of a handwritten amount of 1 to 5 digits. The digits come from MNIST. The layout, alterations (fraud cases) and scanner defects are simulated.

**Model output.** One shared network with **5 softmax heads**, one per position. Each head predicts 11 classes: digits 0-9, plus class 10 for an empty position. The amount is read by combining the 5 heads, and the **confidence** is the product of the 5 maximum probabilities.

**Business question.** Which share of cheques can be processed automatically at 99% precision?

| Official targets | Value |
|---|---|
| Exact amount accuracy | ≥ 0.85 |
| Automation rate at 99% precision | ≥ 0.40 |

---

## Key Results (test set, evaluated once)

| Metric | MLP (final) | CNN (final) | Target |
|---|---|---|---|
| Exact amount accuracy | 0.118 | **0.537** | 0.85 |
| Automation rate @ 99% | 0.010 | **0.164** | 0.40 |
| Precision on the 10% most confident cheques | 0.600 | **0.994** | n/a |
| Confidence AUC | 0.875 | 0.880 | n/a |
| ECE (calibration error) | **0.012** | 0.060 | n/a |
| Parameters | 2,268,983 | 587,255 | n/a |

- **CNN − MLP on the test set: +0.419** exact accuracy (paired bootstrap 95% CI: [+0.405 ; +0.433]).
- Across 3 random seeds on validation, the CNN scores 0.536 ± 0.013 and the MLP 0.114 ± 0.005.
- **Neither model reaches the targets.** The CNN is clearly better but still far from 0.85 and 0.40. These gains are measured on this simulated dataset only.

### Main takeaways

1. **The reference MLP overfits badly and is overconfident.** Its exact accuracy is 0.35 on train but only 0.07 on validation, with ECE = 0.37.
2. **Regularisation fixes calibration more than accuracy.** Dropout, early stopping and hyperparameter tuning (14 configurations) brought ECE down to about 0.01 but moved accuracy by only about 1 point.
3. **Shift augmentation (±4 px) gave the largest MLP gain** (about +2.5 points), because the MLP has a separate weight for every pixel.
4. **Fixing the digit position makes reading much easier.** On a cropped units digit, an isolated MLP reaches ~0.88-0.90 versus ~0.66 for the units head of the 5-head model.
5. **Weight sharing and local filters matter.** Replacing flatten + dense with 3×3 convolutions multiplies exact accuracy by about 4.5 with fewer parameters.
6. **The confidence score is partly a proxy for amount length.** Short amounts are easier and get higher confidence, so the AUC drops at equal length.
7. **The 99%-precision thresholds do not transfer well.** A threshold fixed on validation gave about 93-94% real precision on test, because the top of the ranking contains very few cheques.

---

## Notebook Structure

| # | Section |
|---|---|
| 1 | Imports and random seed |
| 2 | Data loading and exploration |
| 3 | Utility functions (decoding, confidence, calibration, automation curve) |
| 4 | Reference MLP architecture |
| 5 / 5b | Reference training, metrics, and alternative confidence scores |
| 6 | Overfitting and overconfidence: causes and remedies |
| 7 | Hyperparameter sweep and model selection |
| 8 | Shift augmentation and theoretical checks (parameter count, cross-entropy gradient) |
| 9 | Final MLP and single test evaluation |
| 10 | Why the MLP stays limited (4 hypotheses + isolated-digit experiment) |
| 11 | CNN comparison (training, robustness, seeds, final test) |

---

## Methodology Highlights

- **Strict protocol.** All choices (architecture, epoch, augmentation, temperature, thresholds) are made on **validation**. The test set is used **once per model**, and guard cells prevent re-running it in the same session.
- **Reproducibility.** Fixed seed (`SEED = 0`) and library versions printed at the start.
- **Experiment log.** Every experiment, failures included, is written to `journal.jsonl`.
- **Automatic selection rules.** The epoch and the configuration are chosen by code, preferring models within one standard error of the best.
- **Honest uncertainty.** Sampling error bars, bootstrap confidence intervals, and a "prudent" automation rate based on a Wilson lower bound.
- **Calibration.** Temperature scaling, with a split-half check (fit on one half of validation, evaluate on the other).
- **Baselines.** Per-position results are always compared to the constant answer (most frequent class), because empty positions inflate accuracy.

### Architectures

**MLP (final):** `Flatten → Dense 512 → Dense 256 → Dense 128 (dropout 0.3) → 5 × Dense(11, softmax)`, Adam 1e-3, shifts ±4 px, 80 epochs.

**CNN:** `[Conv 3×3 + ReLU + MaxPool 2×2] × 3 (32, 64, 64 filters) → Flatten → Dense 128 → Dropout 0.3 → 5 × Dense(11, softmax)`. There is no global pooling, because each head needs to know where its digit is.

---

## Getting Started

### Option 1: Google Colab (recommended)

1. Open the notebook in Colab :
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hafssa-MI/cheque-amount-reading-mlp-vs-cnn/blob/main/Jalon_1_MLP_et_CNN_corrige.ipynb)
2. Select a **GPU runtime** (Runtime → Change runtime type → T4 GPU).
3. Run *Runtime → Restart session and run all*. When prompted, upload `uc1_domaineA.npz`.

> Full execution takes **several tens of minutes** on a GPU. The test-set cells can only run once per session.

### Option 2: Local

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install tensorflow==2.20.0 scikit-learn numpy matplotlib pandas jupyter
jupyter notebook Jalon_1_MLP_et_CNN_corrige.ipynb
```

Replace the Colab upload cell (`files.upload()`) with a local path to `uc1_domaineA.npz`.

### Dataset

The dataset `uc1_domaineA.npz` is provided by the course and is **not included in this repository**. It contains 12 arrays:

| Key | Content |
|---|---|
| `X_train / X_val / X_test` | 32 × 128 grayscale images (15,000 / 3,000 / 5,000) |
| `montant_*` | Ground-truth amount (string of 1 to 5 digits) |
| `altere_*` | Alteration flag (0 = normal, 1 = altered/fraud) |
| `type_alteration_*` | Integer code of the alteration type |

A second scanner domain (scanner B) is part of the case study but is **not used in this milestone**.

---

## Repository Structure

```text
.
├── Jalon_1_MLP_et_CNN_corrige.ipynb   # Main notebook (in French)
├── README.md                       
└── journal.jsonl                      # Experiment log (optional)
```

---

## Screenshots

<!-- Replace with your own exported figures -->

## Screenshots

### Data exploration
<img src="https://github.com/user-attachments/assets/7a7c524a-f572-40a4-b331-e739034ad348" alt="Data exploration" width="800">

*Training examples from the simulated Banque Atlas cheques (scanner A).*

### MLP training curves (reference model)
<img src="https://github.com/user-attachments/assets/d520b2b5-b173-404a-a4eb-f4cc2f7c6dbf" alt="MLP training curves" width="700">

*Training and validation diverge quickly: the reference MLP overfits.*

### Automation vs precision (MLP vs CNN)
<img src="https://github.com/user-attachments/assets/7cfe9199-667e-42e1-b03d-edf139962663" alt="Automation vs precision, MLP vs CNN" width="800">

*Share of cheques processed automatically against reading precision, on the test set.*

### MLP vs CNN: accuracy by position and by amount length
<img src="https://github.com/user-attachments/assets/aa583bf2-4b11-4a11-836a-35df96433311" alt="MLP vs CNN by position and length" width="800">

*The CNN improves every position and holds up much better on long amounts.*
---

## Tech Stack

Python · TensorFlow / Keras · NumPy · pandas · scikit-learn · Matplotlib · Google Colab (T4 GPU)

---

## Limitations

- One seed per model in the main comparison (3 seeds only in the optional section 11.9).
- The MLP received a sweep of 14 configurations, while the CNN received only 4 trials.
- The data is partly simulated, so conclusions may not transfer to real cheques.
- Hypotheses on why the MLP fails are supported by experiments, not proven.

---

## Authors

- **Hafssa Miftah Idrissi** · [GitHub](https://github.com/Hafssa-MI) · [LinkedIn](https://www.linkedin.com/in/hafssa-miftah-idrissi-5537a8319/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B1hAi42Z6SLCeYXS5PIyPRg%3D%3D)


---

## License

Released for educational purposes.
