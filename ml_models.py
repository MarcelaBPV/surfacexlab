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
