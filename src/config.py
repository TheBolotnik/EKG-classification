from __future__ import annotations
import os
import yaml
from typing import Optional

CONFIG_NAME = "ekg_config.yaml"


def project_root() -> str:
    # .../src/config.py -> .../
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def config_path() -> str:
    return os.path.join(project_root(), CONFIG_NAME)


def load_config() -> dict:
    p = config_path()
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg: dict) -> None:
    with open(config_path(), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=True)


def find_ptbxl_candidates() -> list[str]:
    home = os.path.expanduser("~")
    candidates = [
        os.getenv("PTBXL_PATH"),
        os.path.join(project_root(), "data", "ptb-xl", "1.0.1"),
        os.path.join(project_root(), "data", "ptbxl", "1.0.1"),
        os.path.join(project_root(), "ptb-xl", "1.0.1"),
        os.path.join(home, "datasets", "PTB-XL", "1.0.1"),
        os.path.join(home, "data", "PTB-XL", "1.0.1"),
        os.path.join(home, "ptb-xl", "1.0.1"),
    ]
    return [p for p in candidates if p and os.path.exists(p)]


def resolve_base_path(interactive: bool = True) -> Optional[str]:
    """
    Возвращает корректный base_path:
    1) ENV PTBXL_PATH
    2) ekg_config.yaml: {base_path: ...}
    3) распространённые каталоги
    4) интерактивно спросить у пользователя и сохранить в ekg_config.yaml
    """
    env_p = os.getenv("PTBXL_PATH")
    if env_p and os.path.exists(env_p):
        return env_p

    cfg = load_config()
    if "base_path" in cfg and os.path.exists(cfg["base_path"]):
        return cfg["base_path"]

    found = find_ptbxl_candidates()
    if found:
        cfg["base_path"] = found[0]
        save_config(cfg)
        return found[0]

    if interactive:
        print("Не найден путь к PTB-XL. Укажите каталог, где лежат ptbxl_database.csv / scp_statements.csv и папка records500/")
        print("Пример: /path/to/ptb-xl/1.0.1")
        try:
            user = input("Введите путь (или оставьте пустым для отмены): ").strip()
        except EOFError:
            user = ""
        if user and os.path.exists(user):
            cfg["base_path"] = user
            save_config(cfg)
            return user

    return None
