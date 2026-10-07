import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate

ROLLNO = "BT2024029"
DEGREE = 10

train = pd.read_csv(f"{ROLLNO}_train_var2.csv")
test = pd.read_csv(f"{ROLLNO}_test_var2.csv")

X_train = train.drop("y", axis=1)
y_train = train["y"]
X_test = test[X_train.columns]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 65)
print("VAR2 - RIDGE POLYNOMIAL REGRESSION")
print("=" * 65)
print(f"Training samples : {len(train)}")
print(f"Test samples     : {len(test)}")
print(f"Input features   : {X_train.shape[1]}")
print(f"Polynomial degree: {DEGREE}")
print("Selecting best alpha using 5-Fold CV...")

# Tune alpha automatically
model = make_pipeline(
    PolynomialFeatures(DEGREE, include_bias=False),
    StandardScaler(),
    RidgeCV(
        alphas=np.logspace(-5, 4, 100),
        cv=cv,
        scoring="neg_mean_squared_error"
    )
)

model.fit(X_train, y_train)

ridge = model.named_steps["ridgecv"]
alpha = ridge.alpha_
terms = model.named_steps["polynomialfeatures"].n_output_features_

# Evaluate the final selected alpha using the same 5 folds
final_model = make_pipeline(
    PolynomialFeatures(DEGREE, include_bias=False),
    StandardScaler(),
    Ridge(alpha=alpha)
)

scores = cross_validate(
    final_model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "mse": "neg_mean_squared_error",
        "r2": "r2"
    },
    n_jobs=-1
)

cv_mse = -scores["test_mse"].mean()
cv_r2 = scores["test_r2"].mean()

print("\n" + "-" * 65)
print("FINAL MODEL RESULTS")
print("-" * 65)
print(f"Model           : Ridge")
print(f"Degree          : {DEGREE}")
print(f"Polynomial terms: {terms}")
print(f"CV MSE          : {cv_mse:.8f}")
print(f"CV R2           : {cv_r2:.8f}")
print(f"Alpha           : {alpha:.8f}")
print("-" * 65)

# Train final model on all training data
print("\nTraining final model on all training data...")

final_model.fit(X_train, y_train)

print("Generating predictions...")

predictions = final_model.predict(X_test)

output_file = f"{ROLLNO}_pred_var2.csv"

pd.DataFrame({"y": predictions}).to_csv(
    output_file,
    index=False
)

print(f"Saved: {output_file}")
print(f"Predictions: {len(predictions)}")
print("=" * 65)