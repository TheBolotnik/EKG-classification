#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys

from src.config import resolve_base_path
from src.dataio import load_metadata, ensure_files_exist
from src.features import scp_to_superclass, build_features
from src.model import balance_undersample, train_xgb, save_model


def main():
    p = argparse.ArgumentParser(description="Train XGBoost ECG classifier on PTB-XL (12 leads).")
    p.add_argument("--base-path", default=None, help="Путь к PTB-XL 1.0.1. Если не указан — возьмём из ekg_config.yaml/ENV.")
    p.add_argument("--leads", type=int, default=None, help="Сколько отведений брать (по умолчанию все).")
    p.add_argument("--out", default="models/xgb.pkl", help="Путь для сохранения модели (joblib).")
    args = p.parse_args()

    base_path = args.base_path or resolve_base_path(interactive=True)
    if not base_path:
        print("Не удалось определить путь к PTB-XL.")
        sys.exit(1)

    print(f"Используем base_path: {base_path}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[1/5] Load metadata...")
    db, scp = load_metadata(base_path)

    print("[2/5] Map SCP codes -> diagnostic_superclass...")
    meta = scp_to_superclass(db, scp)

    print("[3/5] Ensure WFDB files exist...")
    meta = ensure_files_exist(meta, base_path)
    print(f"  usable records: {len(meta)}")

    print("[4/5] Build features...")
    X, y = build_features(meta, base_path, leads=args.leads)
    print(f"  X: {X.shape}, classes: {y.value_counts().to_dict()}")

    print("[5/5] Balance, train, evaluate...")
    Xb, yb = balance_undersample(X, y)
    result = train_xgb(Xb, yb)
    result["feature_names"] = list(Xb.columns)   # фиксируем порядок признаков

    # Метрики
    print("\n=== Classification report (macro/weighted F1) ===")
    macro_f1 = result["report"]["macro avg"]["f1-score"]
    weighted_f1 = result["report"]["weighted avg"]["f1-score"]
    print(f"Macro F1: {macro_f1:.3f}  |  Weighted F1: {weighted_f1:.3f}")
    print("\nPer-class F1:")
    for c in result["classes"]:
        print(f"  {c}: {result['report'][c]['f1-score']:.3f}")

    save_model(result, args.out)
    print(f"\n Модель сохранена: {args.out}")


if __name__ == "__main__":
    main()
