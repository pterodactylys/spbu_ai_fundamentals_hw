from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_common import prepare_dataset, split_data

USE_LOG_TARGET = True

MODEL_SPECS = [
    ("Ridge + TF-IDF", "best_model_ridge_title_tfidf.joblib", "sklearn"),
    ("Дерево решений", "best_tree_model_decision_tree.joblib", "sklearn"),
    ("Случайный лес", "best_tree_model_random_forest.joblib", "sklearn"),
    ("CatBoost", "best_boosting_model_catboost.joblib", "boosting"),
    ("LightGBM", "best_boosting_model_lightgbm.joblib", "boosting"),
]


def _predict_sklearn(pipeline, X: pd.DataFrame) -> np.ndarray:
    pred = pipeline.predict(X)
    if USE_LOG_TARGET:
        return np.expm1(np.clip(pred, 0, 15))
    return np.clip(pred, 0, None)


def _predict_boosting(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    X_feat = bundle["features"].transform(X)
    pred = bundle["model"].predict(X_feat)
    if bundle.get("use_log_target", USE_LOG_TARGET):
        return np.expm1(np.clip(pred, 0, 15))
    return np.clip(pred, 0, None)


def load_available_models(artifacts_dir: Path) -> dict[str, object]:
    loaded = {}
    for label, filename, kind in MODEL_SPECS:
        path = artifacts_dir / filename
        if path.exists():
            loaded[label] = (kind, joblib.load(path))
    return loaded


def build_holdout_split(
    data_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X, y = prepare_dataset(data_path)
    return split_data(X, y, test_size=test_size, random_state=random_state)


def predict_all_models(
    models: dict[str, object],
    X: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for label, (kind, artifact) in models.items():
        if kind == "sklearn":
            out[label] = _predict_sklearn(artifact, X)
        else:
            out[label] = _predict_boosting(artifact, X)
    return out.round(0).astype(int)


def build_comparison_from_models(
    models: dict[str, tuple[str, object]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    preds = predict_all_models(models, X_test)
    table = X_test[["title_only", "brand", "condition", "storage_gb", "iphone_gen"]].copy()
    table["price_true"] = y_test.values.astype(int)
    for col in preds.columns:
        table[col] = preds[col].values
    table["abs_err_mean"] = table[list(preds.columns)].sub(
        table["price_true"], axis=0,
    ).abs().mean(axis=1).astype(int)
    return table.reset_index(drop=True)


def build_comparison_table(
    data_path: Path,
    artifacts_dir: Path,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = build_holdout_split(
        data_path, random_state=random_state,
    )
    models = load_available_models(artifacts_dir)
    if not models:
        raise FileNotFoundError(f"Нет моделей в {artifacts_dir}")

    preds = predict_all_models(models, X_test)
    table = X_test[["title_only", "brand", "condition", "storage_gb", "iphone_gen"]].copy()
    table["price_true"] = y_test.values.astype(int)
    for col in preds.columns:
        table[col] = preds[col].values

    table["abs_err_mean"] = table[list(preds.columns)].sub(
        table["price_true"], axis=0,
    ).abs().mean(axis=1).astype(int)

    return table.reset_index(drop=True)


def listing_meta(table: pd.DataFrame, row_idx: int) -> pd.Series:
    row = table.iloc[row_idx]
    return pd.Series({
        "название": row["title_only"],
        "бренд": row["brand"],
        "состояние": row["condition"],
        "память, ГБ": row.get("storage_gb"),
        "iPhone gen": row.get("iphone_gen"),
        "факт, ₽": int(row["price_true"]),
    })


def show_listing_comparison(
    table: pd.DataFrame,
    row_idx: int,
    model_cols: list[str] | None = None,
) -> pd.DataFrame:
    row = table.iloc[row_idx]
    model_cols = model_cols or [c for c in table.columns if c not in {
        "title_only", "brand", "condition", "storage_gb", "iphone_gen",
        "price_true", "abs_err_mean",
    }]

    rows = [{"источник": "Факт (Avito)", "цена, ₽": int(row["price_true"]), "ошибка, ₽": 0}]
    for name in model_cols:
        pred = int(row[name])
        err = pred - int(row["price_true"])
        rows.append({
            "источник": name,
            "цена, ₽": pred,
            "ошибка, ₽": err,
        })

    return pd.DataFrame(rows)


def plot_listing_comparison(
    comparison: pd.DataFrame,
    *,
    title: str = "",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2ecc71" if s == "Факт (Avito)" else "#3498db" for s in comparison["источник"]]
    bars = ax.bar(comparison["источник"], comparison["цена, ₽"], color=colors, edgecolor="white")
    ax.axhline(comparison.loc[0, "цена, ₽"], color="green", ls="--", lw=1, alpha=0.7, label="факт")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_ylabel("Цена, ₽")
    ax.set_title(title or "Сравнение моделей для объявления")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()
