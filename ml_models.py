# ml_models.py
# -*- coding: utf-8 -*-

from typing import Tuple, Dict
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    r2_score, mean_absolute_error
)


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

    task:
        - "classification"
        - "regression"
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def temporal_pca(
    df: pd.DataFrame,
    feature_cols: list,
    sample_col: str = "sample_code",
    time_col: str = "created_at",
    n_components: int = 2,
) -> Dict:
    """
    PCA considerando múltiplas amostras ao longo do tempo.

    Retorna:
    - df_pca: PCs + sample + tempo
    - explained_variance
    - pca_model
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
