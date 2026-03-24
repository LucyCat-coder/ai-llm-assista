"""Загрузчики документов для RAG (MVP).

Поддержка:
- .txt
- .pdf (через pypdf)

В дипломе можно добавить:
- docx
- html
- csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from pypdf import PdfReader


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_documents_from_folder(folder: Path) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for p in sorted(folder.glob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".txt":
            docs[p.stem] = load_txt(p)
        elif p.suffix.lower() == ".pdf":
            docs[p.stem] = load_pdf(p)
    return docs
