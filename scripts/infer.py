from __future__ import annotations
import argparse
import os
import json
import pandas as pd
import wfdb

from src.config import resolve_base_path
from src.model import load_model
from src.features import extract_stats_per_lead


def build_feature_row_from_record(record_path: str, leads: int | None = None) -> pd.DataFrame:
    if not (os.path.isfile(record_path + ".hea") and os.path.isfile(record_path + ".dat")):
        raise FileNotFoundError(f"WFDB файлы не найдены: {record_path}.hea/.dat")
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal
    if leads is not None:
        if sig.shape[1] < leads:
            raise ValueError(f"В записи {record_path} только {sig.shape[1]} отведений, а запрошено {leads}.")
        sig = sig[:, :leads]
    feats = extract_stats_per_lead(sig)
    return pd.DataFrame([feats])


def align_features_to_training(X: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    feature_names = bundle.get("feature_names")
    if feature_names is None:
        return X
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    return X[feature_names]


def main():
    ap = argparse.ArgumentParser(description="Inference: predict ECG class for a WFDB record.")
    ap.add_argument("--record", required=True, help="Путь к записи WFDB (без .hea/.dat). Может быть относительным от base_path.")
    ap.add_argument("--model", default="models/xgb.pkl", help="Путь к сохранённой модели (joblib).")
    ap.add_argument("--leads", type=int, default=None, help="Сколько отведений брать (как при обучении).")
    ap.add_argument("--json", action="store_true", help="Выводить результат в JSON.")
    ap.add_argument("--proba", action="store_true", help="Показывать вероятности по классам.")
    args = ap.parse_args()

    record = args.record
    if not (os.path.isfile(record + ".hea") and os.path.isfile(record + ".dat")):
        base = resolve_base_path(interactive=False)
        if base:
            candidate = os.path.join(base, record)
            if os.path.isfile(candidate + ".hea") and os.path.isfile(candidate + ".dat"):
                record = candidate

    bundle = load_model(args.model)
    model = bundle["model"]
    le = bundle["label_encoder"]
    classes = list(bundle["classes"])

    X_one = build_feature_row_from_record(record, leads=args.leads)
    X_one = align_features_to_training(X_one, bundle)

    try:
        model.set_params(device="cpu")
    except Exception:
        pass

    y_pred = model.predict(X_one)[0]
    label = le.inverse_transform([y_pred])[0]

    result = {"label": str(label), "record": record, "n_features": int(X_one.shape[1])}

    if args.proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_one)[0]
        result["proba"] = {cls: float(p) for cls, p in zip(classes, proba)}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Record: {record}")
        print(f"Predicted: {result['label']}  |  n_features: {result['n_features']}")
        if "proba" in result:
            print("Probabilities:")
            for cls, p in result["proba"].items():
                print(f"  {cls}: {p:.3f}")


if __name__ == "__main__":
    main()
