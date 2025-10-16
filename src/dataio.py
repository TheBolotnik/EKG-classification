from __future__ import annotations
import os
import pandas as pd


def load_metadata(base_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает метаданные PTB-XL:
      - ptbxl_database.csv
      - scp_statements.csv
    """
    db = pd.read_csv(os.path.join(base_path, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(base_path, "scp_statements.csv"))
    return db, scp


def ensure_files_exist(df: pd.DataFrame, base_path: str) -> pd.DataFrame:
    """
    Фильтрует записи, оставляя только те, для которых существуют WFDB файлы (.hea и .dat).
    """
    keep = []
    for idx, row in df.iterrows():
        p = os.path.join(base_path, row["filename_lr"])
        if os.path.isfile(p + ".hea") and os.path.isfile(p + ".dat"):
            keep.append(idx)
    return df.loc[keep].reset_index(drop=True)
