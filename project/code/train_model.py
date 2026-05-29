from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ml_common import evaluate_model, make_preprocessor, prepare_dataset, split_data

TABULAR_NUMERIC = [
    "ram_gb", "storage_gb", "iphone_gen", "photos_count", "title_len",
    "description_len", "days_since_posted", "is_pro", "is_max", "is_mini",
    "is_plus", "is_promoted", "has_warranty", "has_trade_in", "has_box", "has_charger",
]
TABULAR_CATEGORICAL = ["brand", "condition", "district"]


@dataclass
class Paths:
    input_path: Path
    output_dir: Path


def train(paths: Paths, test_size: float = 0.2, random_state: int = 42, use_log_target: bool = True) -> None:
    X, y = prepare_dataset(paths.input_path)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state,
    )

    candidates: list[tuple[str, Pipeline]] = [
        ("dummy_mean", Pipeline([("model", DummyRegressor(strategy="mean"))])),
        (
            "ridge_tabular",
            Pipeline([
                ("preprocess", make_preprocessor(include_text=False, scale_numeric=True)),
                ("model", Ridge(alpha=5.0, random_state=random_state)),
            ]),
        ),
        (
            "ridge_title_tfidf",
            Pipeline([
                ("preprocess", make_preprocessor(include_text=True, text_column="title_only", scale_numeric=True)),
                ("model", Ridge(alpha=3.0, random_state=random_state)),
            ]),
        ),
        (
            "ridge_full_text",
            Pipeline([
                ("preprocess", make_preprocessor(include_text=True, text_column="text", scale_numeric=True)),
                ("model", Ridge(alpha=2.0, random_state=random_state)),
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
            "numeric": "median impute + StandardScaler",
            "categorical": "most_frequent impute + OneHotEncoder(max_categories=30)",
            "text": "TfidfVectorizer on title or title+description (800 features, min_df=2)",
            "tabular_engineered": TABULAR_NUMERIC + TABULAR_CATEGORICAL,
        },
        "models": results,
        "best_model": best["model"],
    }

    metrics_path = paths.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    model_path = paths.output_dir / f"best_model_{best['model']}.joblib"
    joblib.dump(best["fitted_model"], model_path)

    preds_path = paths.output_dir / f"predictions_{best['model']}.csv"
    preds_df = X_test.copy()
    preds_df["price_true"] = y_test.values
    preds_df["price_pred"] = best["predictions"]
    preds_df.to_csv(preds_path, index=False)

    print(f"Rows after cleaning: {len(X)}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved model: {model_path}")
    print(f"Best model: {best['model']}")
    print(f"MAPE: {best['mape']:.2%}  MAE: {best['mae']:.0f} ₽  RMSE: {best['rmse']:.0f} ₽")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Ridge regression for Avito smartphone prices")
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
