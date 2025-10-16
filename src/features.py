from __future__ import annotations
import ast
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import wfdb


def scp_to_superclass(df_meta: pd.DataFrame, df_scp: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует scp_codes (словарь кодов в строке) в диагностический суперкласс.
    Суперклассы: NORM, MI, STTC, CD, HYP.
    """
    scp_diag = df_scp[df_scp["diagnostic"] == 1][["Unnamed: 0", "diagnostic_class"]].copy()
    scp_diag.columns = ["scp_code", "diagnostic_class"]
    scp_map = scp_diag.set_index("scp_code")["diagnostic_class"].to_dict()

    def to_super(d):
        if isinstance(d, str):
            d = ast.literal_eval(d)  # безопаснее, чем eval
        classes = list({scp_map[c] for c in d.keys() if c in scp_map})
        return classes[0] if classes else None

    out = df_meta.copy()
    out["diagnostic_superclass"] = out["scp_codes"].apply(to_super)
    out = out[out["diagnostic_superclass"].notna()].reset_index(drop=True)
    return out


def extract_stats_per_lead(signal: np.ndarray) -> dict:
    """
    Расчёт 7 статистических признаков для каждого отведения.
    signal: shape (T, L) — T отсчётов, L — число отведений.
    """
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
            f"energy_{lead}": float(np.sum(x ** 2)),
        })
    return feats


def build_features(df_meta: pd.DataFrame, base_path: str, leads: int | None = None):
    """
    Читает WFDB записи и строит матрицу признаков X и метки y.
    Если leads задано, ограничивает числом отведений (например, 3 -> I, II, III).
    По умолчанию берёт все доступные отведения.
    """
    records, labels = [], []
    for _, row in df_meta.iterrows():
        rec_path = os.path.join(base_path, row["filename_lr"])
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal
        if leads is not None and sig.shape[1] < leads:
            continue
        feats = extract_stats_per_lead(sig if leads is None else sig[:, :leads])
        records.append(feats)
        labels.append(row["diagnostic_superclass"])
    X = pd.DataFrame(records)
    y = pd.Series(labels, name="label")
    return X, y
