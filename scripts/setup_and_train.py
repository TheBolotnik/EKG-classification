from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable
import requests
import pandas as pd

from src.config import save_config, project_root
from src.dataio import load_metadata, ensure_files_exist
from src.features import scp_to_superclass, build_features
from src.model import balance_undersample, train_xgb, save_model

PHYSIONET_BASE = "https://physionet.org/files/ptb-xl/1.0.1"
CSV_FILES = ["ptbxl_database.csv", "scp_statements.csv"]


# --- НОВЫЙ БЛОК: быстрые параллельные загрузки ---
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

PHYSIONET_BASE = "https://physionet.org/files/ptb-xl/1.0.1"
CSV_FILES = ["ptbxl_database.csv", "scp_statements.csv"]

def _requests_session():
    """Session с пулом подключений и ретраями."""
    sess = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess

def download_file(url: str, dest: Path, chunk: int = 1 << 19, timeout: int = 60):
    """
    Скачка одного файла с прогрессом (если есть Content-Length).
    chunk=512KB (крупнее, чем обычно) — ускоряет.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = _requests_session()
    with sess.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        t0 = time.time()
        if total > 0:
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
            ) as pbar:
                for blob in r.iter_content(chunk_size=chunk):
                    if blob:
                        f.write(blob)
                        got += len(blob)
                        pbar.update(len(blob))
        else:
            with open(dest, "wb") as f:
                for blob in r.iter_content(chunk_size=chunk):
                    if blob:
                        f.write(blob)
                        got += len(blob)
    dt = time.time() - t0
    # короткая строка-лог:
    print(f"✓ {dest.name}: {got/1e6:.1f} MB за {dt:.1f}s")

def _download_one(url: str, dest: Path, retries: int = 3) -> tuple[str, bool, str]:
    """Рабочая функция для пула потоков: возвращает (имя, ok, err)."""
    if dest.exists():
        return (dest.name, True, "exists")
    try:
        download_file(url, dest)
        return (dest.name, True, "")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        return (dest.name, False, err)

def parallel_download(pairs: list[tuple[str, Path]], max_workers: int = 16):
    """
    Параллельная скачка списка файлов.
    pairs: [(url, Path), ...]
    """
    if not pairs:
        return
    # уберём уже существующие
    pairs = [(u, p) for (u, p) in pairs if not p.exists()]
    if not pairs:
        print("✓ Всё уже скачано.")
        return

    print(f"↓ Скачиваю {len(pairs)} файлов, потоков: {max_workers}")
    ok_cnt, fail_cnt = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(pairs), desc="FILES", unit="file") as pbar:
        futures = {ex.submit(_download_one, url, dest): (url, dest) for url, dest in pairs}
        for fut in as_completed(futures):
            name, ok, err = fut.result()
            if ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                # можно логировать подробно:
                # print(f"✗ {name} -> {err}")
            pbar.update(1)
    if fail_cnt:
        print(f"⚠️ Удалось скачать: {ok_cnt}, ошибок: {fail_cnt} (повтори запуск — есть ретраи).")
    else:
        print(f"✓ Скачано успешно: {ok_cnt} файлов.")

def ensure_csvs(base_dir: Path):
    pairs = []
    for name in CSV_FILES:
        target = base_dir / name
        if not target.exists():
            url = f"{PHYSIONET_BASE}/{name}"
            pairs.append((url, target))
    parallel_download(pairs, max_workers=int(os.getenv("DL_WORKERS", "8")))
    for name in CSV_FILES:
        target = base_dir / name
        if target.exists():
            print(f"✓ CSV: {target}")

def download_records_subset(base_dir: Path, filenames_lr: Iterable[str], limit: int):
    """
    Параллельно качаем пары .hea/.dat для первых N путей.
    """
    pairs = []
    n = 0
    for rel in filenames_lr:
        if n >= limit:
            break
        for ext in (".hea", ".dat"):
            url = f"{PHYSIONET_BASE}/{rel}{ext}"
            dest = base_dir / f"{rel}{ext}"
            if not dest.exists():
                pairs.append((url, dest))
        n += 1
    workers = int(os.getenv("DL_WORKERS", "16"))
    parallel_download(pairs, max_workers=workers)
    print(f"✓ Готово: {n} записей (по 2 файла на запись).")

def ensure_records(base_dir: Path, mode: str, subset_size: int):
    """
    mode:
      - 'csvs'   : только CSV (без сигналов)
      - 'subset' : CSV + первые subset_size записей WFDB (параллельно)
      - 'full'   : CSV + все записи WFDB (параллельно, ДОЛГО!)
    """
    ensure_csvs(base_dir)

    if mode == "csvs":
        print("Режим 'csvs': сигналы WFDB не скачиваем.")
        return

    db = pd.read_csv(base_dir / "ptbxl_database.csv")

    if mode == "subset":
        filenames = db["filename_lr"].tolist()
        print(f"↓ Подмножество WFDB: {subset_size} записей")
        download_records_subset(base_dir, filenames, subset_size)
        return

    if mode == "full":
        filenames = db["filename_lr"].tolist()
        print(f"⚠️ Полная загрузка WFDB: {len(filenames)} записей — долго и много места.")
        download_records_subset(base_dir, filenames, limit=len(filenames))
        return


def main():
    ap = argparse.ArgumentParser(description="Setup PTB-XL locally (download) and train model.")
    ap.add_argument("--ptbxl-dir", default=str(Path(project_root()) / "data" / "ptb-xl" / "1.0.1"),
                    help="Куда скачивать/искать PTB-XL (по умолчанию ./data/ptb-xl/1.0.1)")
    ap.add_argument("--download", choices=["csvs", "subset", "full"], default="subset",
                    help="Что скачивать: только CSV, подмножество WFDB, или всё целиком")
    ap.add_argument("--subset-size", type=int, default=500, help="Размер подмножества WFDB в режиме subset")
    ap.add_argument("--leads", type=int, default=None, help="Сколько отведений брать (по умолчанию все)")
    ap.add_argument("--out", default="models/xgb.pkl", help="Куда сохранить обученную модель")
    args = ap.parse_args()

    base_dir = Path(args.ptbxl_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1) загрузка данных
    ensure_records(base_dir, mode=args.download, subset_size=args.subset_size)

    # 2) сохраняем путь в конфиг
    cfg = {"base_path": str(base_dir)}
    save_config(cfg)
    print(f"✓ Путь к PTB-XL сохранён в ekg_config.yaml: {cfg['base_path']}")

    # 3) обучение
    print("[1/5] Load metadata...")
    db, scp = load_metadata(str(base_dir))

    print("[2/5] Map SCP codes -> diagnostic_superclass...")
    meta = scp_to_superclass(db, scp)

    print("[3/5] Ensure WFDB files exist...")
    meta = ensure_files_exist(meta, str(base_dir))
    print(f"  usable records: {len(meta)}")
    if len(meta) == 0:
        print("Не найдено доступных WFDB записей. Выбери --download subset/full.")
        sys.exit(1)

    print("[4/5] Build features...")
    X, y = build_features(meta, str(base_dir), leads=args.leads)
    print(f"  X: {X.shape}, classes: {y.value_counts().to_dict()}")

    print("[5/5] Balance, train, evaluate...")
    Xb, yb = balance_undersample(X, y)
    result = train_xgb(Xb, yb)
    result["feature_names"] = list(Xb.columns)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(result, args.out)
    print(f"\n💾 Модель сохранена: {args.out}")
    print("Готово — предобученную модель можно коммитить (лучше через Git LFS или Releases).")


if __name__ == "__main__":
    main()
