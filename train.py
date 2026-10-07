from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

ROLLNO = "BT2024029"
SEED = 42
FOLDS = 5

# Hyperparameter grids
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 0.5 , 1, 10, 100, 1000,]
LASSO_ALPHAS = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

CV = KFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=SEED
)

ROOT = Path(__file__).resolve().parent


def find_file(kind, var):
    exact = ROOT / f"{ROLLNO}_{kind}_var{var}.csv"

    if exact.exists():
        return exact

    files = sorted(
        ROOT.glob(f"{ROLLNO}_{kind}_var{var}*.csv")
    )

    files = [
        f for f in files
        if "pred_" not in f.name.lower()
    ]

    if len(files) == 1:
        return files[0]

    if not files:
        raise FileNotFoundError(
            f"No {kind} file found for var{var}"
        )

    raise RuntimeError(
        f"Multiple {kind} files found for var{var}:\n" +
        "\n".join(f"  {f.name}" for f in files)
    )


def evaluate(model, X, y):
    scores = cross_validate(
        model,
        X,
        y,
        cv=CV,
        scoring={
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=-1
    )

    return (
        -scores["test_mse"].mean(),
        scores["test_r2"].mean()
    )


def search_degree(X, y, degree):

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_poly = poly.fit_transform(X)

    candidates = []

    # --------------------------------------------------
    # OLS
    # --------------------------------------------------

    mse, r2 = evaluate(
        LinearRegression(),
        X_poly,
        y
    )

    candidates.append(
        ("OLS", mse, r2, None)
    )

    # --------------------------------------------------
    # Ridge
    # --------------------------------------------------

    best = None

    for alpha in RIDGE_ALPHAS:

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )

        mse, r2 = evaluate(
            model,
            X_poly,
            y
        )

        if best is None or mse < best[0]:
            best = (mse, r2, alpha)

    candidates.append(
        ("Ridge", best[0], best[1], best[2])
    )

    # --------------------------------------------------
    # Lasso
    # --------------------------------------------------

    best = None

    for alpha in LASSO_ALPHAS:

        model = make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=alpha,
                max_iter=20000,
                tol=1e-3,
                random_state=SEED
            )
        )

        mse, r2 = evaluate(
            model,
            X_poly,
            y
        )

        if best is None or mse < best[0]:
            best = (mse, r2, alpha)

    candidates.append(
        ("Lasso", best[0], best[1], best[2])
    )

    return candidates, X_poly.shape[1]


def solve(var, max_degree):

    train_file = find_file("train", var)

    train = pd.read_csv(train_file)

    features = [
        c for c in train.columns
        if c != "y"
    ]

    X = train[features].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)

    print("\n" + "=" * 80)
    print(f"PROCESSING VAR{var}")
    print("=" * 80)
    print(f"Training file : {train_file.name}")
    print(f"Training rows : {len(train)}")
    print(f"Features      : {len(features)}")
    print(f"Degrees       : 1 to {max_degree}")
    print(f"CV            : {FOLDS}-Fold KFold")
    print()

    overall_best = None

    for degree in range(1, max_degree + 1):

        results, n_features = search_degree(
            X,
            y,
            degree
        )

        print(
            f"Degree {degree:2d} "
            f"(features={n_features:5d})"
        )

        for method, mse, r2, alpha in results:

            extra = ""

            if alpha is not None:
                extra = f", alpha={alpha:g}"

            print(
                f"  {method:<10}"
                f"MSE={mse:.6f}  "
                f"R2={r2:.6f}"
                f"{extra}"
            )

            candidate = (
                mse,
                degree,
                method,
                r2,
                alpha,
                n_features
            )

            if (
                overall_best is None
                or mse < overall_best[0]
            ):
                overall_best = candidate

        print()

    # --------------------------------------------------
    # Final winner
    # --------------------------------------------------

    mse, degree, method, r2, alpha, n_features = (
        overall_best
    )

    print("-" * 80)
    print(f"FINAL WINNER - VAR{var}")
    print("-" * 80)
    print(f"Model           : {method}")
    print(f"Degree          : {degree}")
    print(f"Polynomial terms: {n_features}")
    print(f"CV MSE          : {mse:.8f}")
    print(f"CV R2           : {r2:.8f}")

    if alpha is not None:
        print(f"Alpha           : {alpha:g}")

    print("-" * 80)


def main():

    # Var1: degree 1 to 10
    solve("1", 10)

    # Var2: degree 1 to 20
    solve("2", 20)


if __name__ == "__main__":
    main()