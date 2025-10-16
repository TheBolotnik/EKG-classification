from __future__ import annotations
import joblib
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


def balance_undersample(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Простой undersampling до размера наименьшего класса.
    """
    df = X.copy()
    df["label"] = y.values
    n = df["label"].value_counts().min()
    parts = [
        resample(
            df[df.label == c], replace=False, n_samples=n, random_state=random_state
        )
        for c in df["label"].unique()
    ]
    bal = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return bal.drop(columns="label"), bal["label"]


def train_xgb(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """
    Обучение XGBoost, отчёт по метрикам и выдача артефактов (вместе с LabelEncoder).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=random_state,
        # для GPU в XGBoost 2.x:
        tree_method="hist",
        device="cuda"
    )

    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_te)

    report = classification_report(
        le.inverse_transform(y_te),
        le.inverse_transform(y_pr),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        le.inverse_transform(y_te),
        le.inverse_transform(y_pr),
        labels=le.classes_,
    )

    return {
        "model": clf,
        "label_encoder": le,
        "report": report,
        "confusion_matrix": cm,
        "classes": le.classes_,
    }


def save_model(bundle: dict, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str) -> dict:
    return joblib.load(path)
