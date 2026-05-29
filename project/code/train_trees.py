#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ml_common import evaluate_model, make_preprocessor, prepare_dataset, split_data


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

    preprocess = make_preprocessor(
        include_text=True,
        text_column="title_only",
        scale_numeric=False,
        max_tfidf_features=500,
    )

    candidates: list[tuple[str, Pipeline]] = [
        ("dummy_mean", Pipeline([("model", DummyRegressor(strategy="mean"))])),
        (
            "decision_tree",
            Pipeline([
                ("preprocess", preprocess),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=14,
                        min_samples_leaf=8,
                        min_samples_split=16,
                        random_state=random_state,
                    ),
                ),
            ]),
        ),
        (
            "random_forest",
            Pipeline([
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=10,
                        min_samples_split=20,
                        max_features=0.3,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]),
        ),
    ]

    results = []
    best = None
    best_mape = float("inf")

    for name, pipeline in candidates:
        metrics = evaluate_model(
            name, pipeline, X_train, X_test, y_train, y_test,
            use_log_target=use_log_target,
        )
        results.append({k: v for k, v in metrics.items() if k not in {"predictions", "fitted_model"}})
        print(
            f"{name:16}  MAPE={metrics['mape']:.1%}  "
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
            "numeric": "median impute, без StandardScaler (деревья)",
            "categorical": "OneHotEncoder(max_categories=30)",
            "text": "TfidfVectorizer на title_only (500 features)",
            "models_description": {
                "decision_tree": "max_depth=14, min_samples_leaf=8",
                "random_forest": "n_estimators=200, max_depth=12, min_samples_leaf=10, max_features=0.3",
            },
        },
        "models": results,
        "best_model": best["model"],
    }

    metrics_path = paths.output_dir / "metrics_trees.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    model_path = paths.output_dir / f"best_tree_model_{best['model']}.joblib"
    joblib.dump(best["fitted_model"], model_path)

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
    parser = argparse.ArgumentParser(description="Tree models for Avito smartphone prices")
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
