from __future__ import annotations
import argparse
import os

from src.dataio import load_metadata, ensure_files_exist
from src.features import scp_to_superclass, build_features
from src.model import balance_undersample, train_xgb, save_model, load_model


def main():
    p = argparse.ArgumentParser(description="Fine-tune existing XGBoost model on new ECG data.")
    p.add_argument("--base-path", required=True, help="Путь к каталогу с новыми данными (PTB-XL совместимый).")
    p.add_argument("--pretrained", required=True, help="Путь к существующей модели (bundle).")
    p.add_argument("--mode", choices=["add_trees", "retrain"], default="add_trees",
                   help="add_trees: добавить деревья; retrain: обучить заново на объединённом наборе.")
    p.add_argument("--leads", type=int, default=None, help="Сколько отведений брать (как при обучении).")
    p.add_argument("--out", default="models/xgb_finetuned.pkl", help="Куда сохранить обновлённую модель.")
    p.add_argument("--extra_estimators", type=int, default=100, help="Сколько деревьев добавить в режиме add_trees.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Загружаем предобученную модель
    bundle = load_model(args.pretrained)
    base_model = bundle["model"]
    le = bundle["label_encoder"]

    # Готовим новые данные
    db, scp = load_metadata(args.base_path)
    meta = scp_to_superclass(db, scp)
    meta = ensure_files_exist(meta, args.base_path)
    X_new, y_new = build_features(meta, args.base_path, leads=args.leads)
    Xb, yb = balance_undersample(X_new, y_new)

    if args.mode == "add_trees":
        from xgboost import XGBClassifier
        extra = args.extra_estimators
        new_n = base_model.get_params()["n_estimators"] + extra
        finetuned = XGBClassifier(**{**base_model.get_params(), "n_estimators": new_n})
        finetuned.fit(Xb, le.transform(yb), xgb_model=base_model)

        bundle["model"] = finetuned
        bundle["feature_names"] = list(Xb.columns)

    else:  # retrain
        result = train_xgb(Xb, yb)
        result["feature_names"] = list(Xb.columns)
        # сохраняем порядок классов исходной модели (если совпадают)
        result["label_encoder"] = le
        result["classes"] = bundle["classes"]
        bundle = result

    save_model(bundle, args.out)
    print(f"[OK] Fine-tuned model saved to: {args.out}")


if __name__ == "__main__":
    main()
