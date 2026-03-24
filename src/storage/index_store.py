"""Мини-хранилище индекса (метаданные).

В MVP мы переиндексируем на каждом запуске.
Файл нужен, чтобы студент понимал, где хранить метаданные/настройки.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonIndexStore:
    def __init__(self, path: str = "data/index_meta.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, meta: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
