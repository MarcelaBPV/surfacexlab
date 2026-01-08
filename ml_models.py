# -*- coding: utf-8 -*-

from typing import Dict
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    r2_score, mean_absolute_error
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def train_random_forest_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict:

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
    }


# =========================================================
# REGRESSÃO
# =========================================================
def train_random_forest_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 400,
    random_state: int = 42,
) -> Dict:

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "feature_importance": pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
    }


# =========================================================
# RANDOM FOREST — VALIDAÇÃO CRUZADA (CV)
# =========================================================
def random_forest_cv(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    n_estimators: int = 300,
    cv: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Random Forest com validação cruzada k-fold.
    """

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        scoring = "f1_weighted"

    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        scoring = "r2"

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    model.fit(X, y)

    return {
        "model": model,
        "cv_scores": scores,
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "feature_importance": pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False),
    }


# =========================================================
# PCA MULTI-AMOSTRA TEMPORAL
# =========================================================
def temporal_pca(
    df: pd.DataFrame,
    feature_cols: list,
    sample_col: str = "sample_code",
    time_col: str = "created_at",
    n_components: int = 2,
) -> Dict:
    """
    PCA considerando múltiplas amostras ao longo do tempo.
    """

    df = df.dropna(subset=feature_cols + [sample_col, time_col]).copy()

    X = df[feature_cols].values
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    PCs = pca.fit_transform(Xs)

    df_pca = pd.DataFrame(
        PCs,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df.index
    )

    df_pca[sample_col] = df[sample_col].values
    df_pca[time_col] = pd.to_datetime(df[time_col])

    return {
        "df_pca": df_pca,
        "explained_variance": pca.explained_variance_ratio_,
        "pca_model": pca,
    }


# =========================================================
# PCA ESTATÍSTICO + BIPLOT (PAPER-LEVEL)
# =========================================================
def pca_biplot(
    X: pd.DataFrame,
    n_components: int = 2,
    scale_loadings: float = 3.0,
) -> Dict:
    """
    Executa PCA e gera gráfico BIPLOT (scores + loadings).
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_

    # --------- Plot ---------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        scores[:, 0],
        scores[:, 1],
        color="royalblue",
        s=40,
        alpha=0.75,
        label="Amostras"
    )

    for i, var in enumerate(X.columns):
        ax.arrow(
            0, 0,
            loadings[i, 0] * scale_loadings,
            loadings[i, 1] * scale_loadings,
            color="crimson",
            alpha=0.9,
            head_width=0.04,
            length_includes_head=True
        )
        ax.text(
            loadings[i, 0] * scale_loadings * 1.1,
            loadings[i, 1] * scale_loadings * 1.1,
            var,
            fontsize=9,
            color="crimson"
        )

    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA Biplot — Scores e Loadings")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance": explained,
        "pca_model": pca,
        "figure": fig,
    }
