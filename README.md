# Polynomial Regression — Machine Learning Assignment

**Roll Number:** BT2024029

This project implements and evaluates **Polynomial Regression** with **OLS, Ridge, and Lasso regularization** on two regression datasets.

The objective is to identify suitable polynomial degrees and regularization parameters while avoiding overfitting and numerical instability.

---

## Overview

Two datasets were evaluated:

| Dataset | Description | Input Variables |
|---|---|---:|
| **Var1** | Net Power Score of a steam turbine | 6 |
| **Var2** | Thermal Anomaly Score over a physical space | 3 |

Each dataset contains:

- 1,000 training samples
- 1,000 test samples
- Hidden test targets

Since test targets are unavailable, model performance is evaluated using **5-fold cross-validation on the training data**.

---

## Final Models

The final selected architectures are:

| Dataset | Pipeline | Degree | Regularizer | α | CV MSE | CV R² |
|---|---|---:|---|---:|---:|---:|
| **Var1** | Polynomial Features → StandardScaler → Lasso | 5 | L1 | 0.010481 | 0.325201 | 0.967505 |
| **Var2** | Polynomial Features → StandardScaler → Ridge | 10 | L2 | 0.533669 | 0.251966 | 0.994923 |

The models were retrained on the complete training sets before generating predictions for the hidden test data.

---

## Methodology

The project uses a two-phase model-selection strategy.

### Phase 1 — Broad Architecture Search

`train.py` evaluates:

- OLS
- Ridge
- Lasso
- Polynomial degrees 1–10 for Var1
- Polynomial degrees 1–20 for Var2
- 5-fold cross-validation
- Coarse regularization parameter values

The purpose of this phase is to identify a promising polynomial degree and model family.

### Phase 2 — Fine-Grained Hyperparameter Tuning

After selecting the model family and degree:

- **Var1:** `LassoCV`
  - 50 α values
  - Range: `10^-4` to `10^-1`

- **Var2:** `RidgeCV`
  - 100 α values
  - Range: `10^-5` to `10^4`

This produces the final regularization parameters used for prediction.

---

## Preventing Data Leakage

`StandardScaler` is fitted only on the training portion of each cross-validation fold.

This ensures that validation information is not used during preprocessing.

---

## Polynomial Feature Expansion

Polynomial features generate powers and interaction terms of the original variables.

For `p` input variables and maximum degree `d`, the number of generated terms without the bias term is:

\[
N_{\text{terms}} =
\binom{p+d}{d}-1
\]

The number of features grows rapidly with polynomial degree.

For example:

- Var1, degree 6 → **923 features**
- Each 5-fold training split → **800 samples**

This makes high-degree unregularized OLS unstable, while Ridge and Lasso provide better control over the expanded feature space.

---

## Model Comparison

### Ordinary Least Squares (OLS)

OLS minimizes the residual error without a regularization penalty.

It can become highly unstable when the polynomial feature space becomes large.

### Ridge Regression

Ridge uses an **L2 penalty** to shrink coefficients.

It is useful when many correlated polynomial features contribute to the prediction.

### Lasso Regression

Lasso uses an **L1 penalty**.

It can drive coefficients exactly to zero, producing a sparser representation and performing implicit feature selection.

---

## Key Result

A notable example occurs for **Var1 at degree 6**:

| Model | CV MSE |
|---|---:|
| OLS | 342.70 |
| Ridge | 0.6142 |
| Lasso | 0.3292 |

At this degree, 923 polynomial features are generated while each CV training fold contains only 800 samples. The large feature space causes OLS to become highly unstable, whereas regularized models remain stable.

---