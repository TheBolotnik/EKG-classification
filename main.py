from __future__ import annotations
import os
import sys
import glob
import subprocess
import pandas as pd
import wfdb
import numpy as np
from scipy.stats import skew, kurtosis

from src.config import resolve_base_path
from src.dataio import load_metadata, ensure_files_exist
from src.features import scp_to_superclass, build_features
from src.model import balance_undersample, train_xgb, save_model, load_model
from src.labels import to_russian  # ← добавить импорт наверху рядом с остальными


ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _first_record_path_from_csv(base: str) -> str | None:
    import pandas as pd, os
    csv_path = os.path.join(base, "ptbxl_database.csv")
    if not os.path.isfile(csv_path):
        return None
    db = pd.read_csv(csv_path)
    for rel in db["filename_lr"].tolist():
        hea = os.path.join(base, f"{rel}.hea")
        dat = os.path.join(base, f"{rel}.dat")
        if os.path.isfile(hea) and os.path.isfile(dat):
            return os.path.join(base, rel)  # без расширения
    return None


def _debug_list_some_files(base: str, limit: int = 3):
    pattern = os.path.join(base, "records500", "**", "*_lr.hea")
    files = glob.glob(pattern, recursive=True)
    print(f"[debug] glob нашёл {len(files)} файлов по шаблону {pattern!r}")
    for p in files[:limit]:
        print("  -", p)


def _extract_one(signal: np.ndarray) -> dict:
    """Та же схема признаков, что и при обучении (7 статистик на отведение)."""
    feats = {}
    L = signal.shape[1]
    for lead in range(L):
        x = signal[:, lead]
        feats.update({
            f"mean_{lead}": float(np.mean(x)),
            f"std_{lead}": float(np.std(x)),
            f"max_{lead}": float(np.max(x)),
            f"min_{lead}": float(np.min(x)),
            f"skew_{lead}": float(skew(x)),
            f"kurt_{lead}": float(kurtosis(x)),
            f"energy_{lead}": float(np.sum(x**2)),
        })
    return feats


def _auto_prepare_dataset():
    """
    Автоматически запускает подготовку датасета:
    эквивалент `python scripts/setup_and_train.py --download subset --subset-size 500`
    """
    setup_script = os.path.join(ROOT, "scripts", "setup_and_train.py")
    if not os.path.isfile(setup_script):
        print("Не найден scripts/setup_and_train.py — не могу автоматически подготовить PTB-XL.")
        return False

    cmd = [sys.executable, setup_script, "--download", "subset", "--subset-size", "500"]
    print("Путь к PTB-XL не найден. Запускаю авто-настройку данных:")
    print("   ", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=ROOT, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Автонастройка завершилась с ошибкой (code={e.returncode}).")
        return False


def main():
    # 1) ищем путь к PTB-XL (без интерактива — полностью автоматический режим)
    base = resolve_base_path(interactive=False)

    # 2) если не нашли — автоматическая загрузка CSV + subset WFDB + обучение
    if not base:
        ok = _auto_prepare_dataset()
        if not ok:
            print("Не удалось подготовить PTB-XL автоматически. "
                  "Запусти вручную: python scripts/setup_and_train.py --download subset --subset-size 500")
            sys.exit(1)
        # после успешной настройки путь уже записан в ekg_config.yaml
        base = resolve_base_path(interactive=False)

    if not base:
        print("Путь к PTB-XL всё ещё не определён. Проверь ekg_config.yaml или переменную PTBXL_PATH.")
        sys.exit(1)

    print(f"base_path: {base}")

    # 3) если модели нет — обучаем «здесь и сейчас»
    model_path = os.path.join(ROOT, "models", "xgb.pkl")
    if not os.path.isfile(model_path):
        print("Модель не найдена — обучаем...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # [1] Метаданные
        db, scp = load_metadata(base)
        # [2] Метки: SCP → суперкласс
        meta = scp_to_superclass(db, scp)
        # [3] Фильтрация по наличию WFDB
        meta = ensure_files_exist(meta, base)
        print(f"  usable records: {len(meta)}")
        if len(meta) == 0:
            print("Нет доступных WFDB записей. Проверь загрузку signals (records500).")
            sys.exit(1)
        # [4] Извлечение признаков (все 12 отведений)
        X, y = build_features(meta, base, leads=None)
        print(f"  X: {X.shape}, classes: {y.value_counts().to_dict()}")
        # [5] Балансировка и обучение
        Xb, yb = balance_undersample(X, y)
        result = train_xgb(Xb, yb)
        result["feature_names"] = list(Xb.columns)
        save_model(result, model_path)
        print(f"Модель сохранена: {model_path}")
    else:
        print("Найдена предобученная модель.")

    # 4) инференс на первой найденной записи
    rec = _first_record_path_from_csv(base)
    if not rec:
        print("Не удалось найти запись через CSV, пробую glob...")
        _debug_list_some_files(base)
        # старый запасной путь через glob (можно оставить как fallback)
        pattern = os.path.join(base, "records500", "**", "*_lr.hea")
        import glob
        files = glob.glob(pattern, recursive=True)
        if files:
            rec = files[0][:-4]

    if not rec:
        print("Не удалось найти *_lr.hea в records500/. Проверь, что данные скачаны полностью.")
        return

    bundle = load_model(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    feature_names = bundle.get("feature_names")

    r = wfdb.rdrecord(rec)
    X_one = pd.DataFrame([_extract_one(r.p_signal)])
    if feature_names:
        for col in feature_names:
            if col not in X_one.columns:
                X_one[col] = 0.0
        X_one = X_one[feature_names]

    y_pred = model.predict(X_one)[0]
    label = le.inverse_transform([y_pred])[0]
    label_ru = to_russian(label)
    print(f"🎉 Предсказанный класс для {rec}: {label} — {label_ru}")

    y_pred = model.predict(X_one)[0]
    label = le.inverse_transform([y_pred])[0]
    print(f"Предсказанный класс для {rec}: {label}")


if __name__ == "__main__":
    main()
