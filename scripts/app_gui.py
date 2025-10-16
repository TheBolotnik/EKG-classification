from __future__ import annotations
import os
import sys
import json
import traceback
import pandas as pd
import wfdb
import tkinter as tk
from tkinter import filedialog, messagebox

# самодостаточный импорт проекта
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import resolve_base_path
from model import load_model
from features import extract_stats_per_lead
from labels import to_russian


def _normalize_record_path(p: str) -> str:
    return p[:-4] if (p.endswith(".hea") or p.endswith(".dat")) else p


def _has_pair(p: str) -> bool:
    return os.path.isfile(p + ".hea") and os.path.isfile(p + ".dat")


def _resolve_full_path(record: str, base_path: str) -> str:
    rec = _normalize_record_path(record)
    if os.path.isabs(rec) and _has_pair(rec):
        return rec
    cand = os.path.join(base_path, rec)
    if _has_pair(cand):
        return cand
    # снять возможные records100/records500 и перебрать
    for prefix in ("records100/", "records500/", "records100\\", "records500\\"):
        if rec.startswith(prefix):
            rec = rec[len(prefix):]
            break
    for head in ("records100", "records500"):
        p = os.path.join(base_path, head, rec)
        if _has_pair(p):
            return p
    raise FileNotFoundError(f"Не найдено пары .hea/.dat для '{record}' (base_path={base_path}).")


def predict_one(rec_path: str, model_path: str, leads: int | None) -> dict:
    base = resolve_base_path(interactive=False)
    if not base:
        raise RuntimeError("base_path не найден. Сначала запусти main.py или scripts/setup_and_train.py.")

    rec = _resolve_full_path(rec_path, base)
    bundle = load_model(model_path)
    model = bundle["model"]
    le = bundle["label_encoder"]
    classes = list(bundle["classes"])
    feature_names = bundle.get("feature_names")

    try:
        model.set_params(device="cpu")  # чтобы не было предупреждений про cuda/cpu mismatch
    except Exception:
        pass

    r = wfdb.rdrecord(rec)
    sig = r.p_signal
    if leads is not None:
        if sig.shape[1] < leads:
            raise ValueError(f"В записи {rec} только {sig.shape[1]} отведений < запрошено {leads}")
        sig = sig[:, :leads]

    feats = extract_stats_per_lead(sig)
    X = pd.DataFrame([feats])
    if feature_names:
        for c in feature_names:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_names]

    y_idx = int(model.predict(X)[0])
    label = le.inverse_transform([y_idx])[0]
    out = {
        "record": rec,
        "label": label,
        "label_ru": to_russian(label),
        "n_features": int(X.shape[1]),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        out["proba"] = {cls: float(p) for cls, p in zip(classes, proba)}
    return out


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EKG Classification (PTB-XL, XGBoost)")
        self.geometry("720x520")
        self.resizable(True, True)

        frm = tk.Frame(self); frm.pack(fill="x", padx=10, pady=10)
        tk.Label(frm, text="Модель (.pkl):").grid(row=0, column=0, sticky="w")
        self.model_entry = tk.Entry(frm, width=70)
        self.model_entry.grid(row=0, column=1, padx=5)
        self.model_entry.insert(0, os.path.join(ROOT, "models", "xgb.pkl"))
        tk.Button(frm, text="Выбрать...", command=self.browse_model).grid(row=0, column=2)

        tk.Label(frm, text="Запись WFDB:").grid(row=1, column=0, sticky="w")
        self.record_entry = tk.Entry(frm, width=70)
        self.record_entry.grid(row=1, column=1, padx=5)
        tk.Button(frm, text="Выбрать .hea...", command=self.browse_record).grid(row=1, column=2)

        tk.Label(frm, text="Отведений (пусто = все):").grid(row=2, column=0, sticky="w")
        self.leads_entry = tk.Entry(frm, width=10); self.leads_entry.grid(row=2, column=1, sticky="w")

        tk.Button(frm, text="Предсказать", command=self.on_predict).grid(row=3, column=1, pady=8, sticky="w")

        self.text = tk.Text(self, wrap="word")
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

    def browse_model(self):
        p = filedialog.askopenfilename(filetypes=[("Joblib/PKL", "*.pkl;*.joblib"), ("All", "*.*")],
                                       initialdir=os.path.join(ROOT, "models"))
        if p:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, p)

    def browse_record(self):
        p = filedialog.askopenfilename(filetypes=[("WFDB header", "*.hea"), ("All", "*.*")])
        if p:
            self.record_entry.delete(0, tk.END)
            self.record_entry.insert(0, p)

    def on_predict(self):
        self.text.delete("1.0", tk.END)
        model = self.model_entry.get().strip()
        record = self.record_entry.get().strip()
        leads_txt = self.leads_entry.get().strip()
        leads = int(leads_txt) if leads_txt else None
        try:
            res = predict_one(record, model, leads)
            self.text.insert(tk.END, f"Record: {res['record']}\n")
            self.text.insert(tk.END, f"Predicted: {res['label']} — {res['label_ru']}\n")
            self.text.insert(tk.END, f"n_features: {res['n_features']}\n")
            if "proba" in res:
                self.text.insert(tk.END, "Probabilities:\n")
                for k, v in res["proba"].items():
                    self.text.insert(tk.END, f"  {k} — {to_russian(k)}: {v:.3f}\n")
        except Exception as e:
            messagebox.showerror("Ошибка", f"{type(e).__name__}: {e}")
            self.text.insert(tk.END, traceback.format_exc())


if __name__ == "__main__":
    App().mainloop()
