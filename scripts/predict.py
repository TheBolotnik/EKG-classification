from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, List, Dict

import pandas as pd
import wfdb

# === Самодостаточный импорт проекта (без необходимости выставлять PYTHONPATH) ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import resolve_base_path             # noqa: E402
from model import load_model                     # noqa: E402
from features import extract_stats_per_lead      # noqa: E402
from labels import to_russian                    # noqa: E402


def _normalize_record_path(p: str) -> str:
    """
    Принимает путь к записи с/без .hea/.dat и возвращает путь БЕЗ расширения.
    """
    if p.endswith(".hea") or p.endswith(".dat"):
        p = p[:-4]
    return p


def _resolve_full_path(record: str, base_path: str) -> str:
    """
    Приводит строку record к реальному пути (БЕЗ расширения), где существуют .hea и .dat.
    Принимает:
      - абсолютный или относительный путь с/без .hea/.дат,
      - укороченный вид '00000/00001_lr',
      - пути, где случайно указано неверное начало (автоподстановка records100/ и records500/).
    """
    def has_pair(p: str) -> bool:
        return os.path.isfile(p + ".hea") and os.path.isfile(p + ".dat")

    rec = _normalize_record_path(record)

    # 1) Абсолютный путь сразу
    if os.path.isabs(rec) and has_pair(rec):
        return rec

    # 2) Относительно base_path
    cand = os.path.join(base_path, rec)
    if has_pair(cand):
        return cand

    # 3) Если пользователь передал только хвост '00000/00001_lr'
    #    или смешал префиксы, попробуем подставить оба варианта
    #    records100/ и records500/
    for prefix in ("records100/", "records500/", "records100\\", "records500\\"):
        if rec.startswith(prefix):
            rec = rec[len(prefix):]
            break

    tails = [os.path.join("records100", rec), os.path.join("records500", rec)]
    for t in tails:
        p = os.path.join(base_path, t)
        if has_pair(p):
            return p

    raise FileNotFoundError(
        f"Не нашёл пару .hea/.dat для '{record}'. "
        f"Попробуй, например: 'records100/00000/00001_lr' (относительно base_path={base_path})."
    )


def _align_features(X: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
    """
    Выравнивает набор признаков под порядок и состав, использованные при обучении.
    Отсутствующие признаки добавляются нулями.
    """
    if not feature_names:
        return X
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    return X[feature_names]


def main():
    ap = argparse.ArgumentParser(description="Predict ECG class for a single PTB-XL WFDB record.")
    ap.add_argument(
        "--input",
        required=True,
        help=(
            "Путь к записи WFDB (без расширения или с .hea/.dat). "
            "Можно указывать относительно base_path из ekg_config.yaml, например: "
            "'records100/00000/00001_lr' или абсолютный путь к .hea/.dat."
        ),
    )
    ap.add_argument(
        "--model",
        default=os.path.join(ROOT, "models", "xgb.pkl"),
        help="Путь к модели (bundle .pkl). По умолчанию: models/xgb.pkl.",
    )
    ap.add_argument(
        "--leads",
        type=int,
        default=None,
        help="Сколько отведений использовать (как при обучении). По умолчанию все 12.",
    )
    ap.add_argument(
        "--proba",
        action="store_true",
        help="Добавить вероятности по классам в вывод.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Выводить результат в формате JSON.",
    )
    args = ap.parse_args()

    # Базовый путь к PTB-XL
    base_path = resolve_base_path(interactive=False)
    if not base_path:
        print("Не найден base_path. Сначала запусти main.py или scripts/setup_and_train.py, либо укажи PTBXL_PATH.")
        sys.exit(1)

    # Резолвим путь записи к реальному .hea/.dat (без расширения)
    rec_path = _resolve_full_path(args.input, base_path)

    # Загрузка бандла модели
    bundle: Dict = load_model(args.model)
    model = bundle["model"]
    le = bundle["label_encoder"]
    classes: List[str] = list(bundle["classes"])
    feature_names: Optional[List[str]] = bundle.get("feature_names")

    # Инференс на CPU, чтобы не было предупреждений о несовпадении устройств
    try:
        model.set_params(device="cpu")
    except Exception:
        pass

    # Чтение сигнала WFDB и извлечение признаков
    rec = wfdb.rdrecord(rec_path)
    sig = rec.p_signal  # shape: (T, L)
    if args.leads is not None:
        if sig.shape[1] < args.leads:
            print(f"В записи {rec_path} только {sig.shape[1]} отведений < запрошено {args.leads}.")
            sys.exit(1)
        sig = sig[:, :args.leads]

    feats = extract_stats_per_lead(sig)
    X = pd.DataFrame([feats])
    X = _align_features(X, feature_names)

    # Предсказание
    y_idx = int(model.predict(X)[0])
    label = le.inverse_transform([y_idx])[0]
    label_ru = to_russian(label)

    result: Dict[str, object] = {
        "record": rec_path,
        "label": str(label),
        "label_ru": str(label_ru),
        "n_features": int(X.shape[1]),
    }

    if args.proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["proba"] = {cls: float(p) for cls, p in zip(classes, proba)}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Record: {result['record']}")
        print(f"Predicted: {result['label']} — {result['label_ru']}  |  n_features: {result['n_features']}")
        if "proba" in result:
            print("Probabilities:")
            for cls, p in result["proba"].items():
                print(f"  {cls} — {to_russian(cls)}: {p:.3f}")


if __name__ == "__main__":
    main()
