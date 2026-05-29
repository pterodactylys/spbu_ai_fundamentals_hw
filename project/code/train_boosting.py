#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

try:
    from lightgbm import LGBMRegressor, early_stopping

    HAS_LIGHTGBM = True
except OSError:
    HAS_LIGHTGBM = False

from ml_common import (
    TABULAR_CATEGORICAL,
    evaluate_boosting,
    fit_boosting_features,
    prepare_dataset,
    split_data,
)


@dataclass
class Paths:
    input_path: Path
    output_dir: Path


def train(
    paths: Paths,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    use_log_target: bool = True,
) -> None:
    X, y = prepare_dataset(paths.input_path)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state,
    )

    features = fit_boosting_features(X_train, max_tfidf_features=300)
    X_tr = features.transform(X_train)
    X_te = features.transform(X_test)
    y_fit = np.log1p(y_train.values) if use_log_target else y_train.values
    y_eval = np.log1p(y_test.values) if use_log_target else y_test.values

    candidates: list[tuple[str, object, dict]] = [
        (
            "catboost",
            CatBoostRegressor(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=0,
                allow_writing_files=False,
            ),
            {
                "eval_set": (X_te, y_eval),
                "cat_features": TABULAR_CATEGORICAL,
                "early_stopping_rounds": 60,
                "use_best_model": True,
            },
        ),
    ]

    if HAS_LIGHTGBM:
        candidates.append(
            (
                "lightgbm",
                LGBMRegressor(
                    n_estimators=1500,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=31,
                    min_child_samples=12,
                    subsample=0.85,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
                {
                    "eval_set": [(X_te, y_eval)],
                    "callbacks": [early_stopping(60, verbose=False)],
                    "categorical_feature": TABULAR_CATEGORICAL,
                },
            ),
        )
    else:
        print("LightGBM недоступен")

    results = []
    best = None
    best_mape = float("inf")

    for name, model, fit_kwargs in candidates:
        metrics = evaluate_boosting(
            name,
            model,
            features,
            X_train,
            X_test,
            y_train,
            y_test,
            use_log_target=use_log_target,
            fit_kwargs=fit_kwargs,
        )
        results.append({k: v for k, v in metrics.items() if k not in {"predictions", "fitted_model", "features"}})
        print(
            f"{name:12}  MAPE={metrics['mape']:.1%}  "
            f"MAE={metrics['mae']:,.0f} ₽  RMSE={metrics['rmse']:,.0f} ₽"
        )
        if metrics["mape"] < best_mape:
            best_mape = metrics["mape"]
            best = metrics

    assert best is not None
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "n_rows": len(X),
        "use_log_target": use_log_target,
        "feature_transforms": {
            "target": "log1p(price)" if use_log_target else "price",
            "numeric": "median impute",
            "categorical": "строковые колонки (CatBoost native / LightGBM categorical)",
            "text": "TfidfVectorizer title_only, 300 features",
            "models_description": {
                "catboost": "depth=6, lr=0.05, early_stopping=60",
                "lightgbm": "max_depth=8, num_leaves=31, early_stopping=60",
            },
        },
        "models": results,
        "best_model": best["model"],
    }

    metrics_path = paths.output_dir / "metrics_boosting.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    bundle = {
        "model": best["fitted_model"],
        "features": best["features"],
        "use_log_target": use_log_target,
        "model_name": best["model"],
    }
    model_path = paths.output_dir / f"best_boosting_model_{best['model']}.joblib"
    joblib.dump(bundle, model_path)

    preds_path = paths.output_dir / f"predictions_{best['model']}.csv"
    preds_df = X_test.copy()
    preds_df["price_true"] = y_test.values
    preds_df["price_pred"] = best["predictions"]
    preds_df.to_csv(preds_path, index=False)

    print("-" * 50)
    print(f"Rows: {len(X)}")
    print(f"Best: {best['model']}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient boosting for Avito smartphone prices")
    parser.add_argument("--input", type=Path, default=Path("../data/smartphones_avito.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-log-target", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        Paths(input_path=args.input, output_dir=args.output_dir),
        test_size=args.test_size,
        random_state=args.random_state,
        use_log_target=not args.no_log_target,
    )


if __name__ == "__main__":
    main()
