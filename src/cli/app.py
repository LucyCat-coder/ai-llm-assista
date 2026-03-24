"""CLI для RAG-шаблона.

Запуск:
    python -m src.cli.app rag_demo --query "Нужен ли фундамент?"

Также можно указать папку с документами:
    python -m src.cli.app rag_demo --docs docs/sample_docs --query "Что такое RAG?"
"""

import argparse
from pathlib import Path

from src.core.rag_pipeline import RagPipeline
from src.parsers.loaders import load_documents_from_folder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-diploma-rag", description="RAG диплом (шаблон)")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("rag_demo", help="Построить индекс и задать вопрос (MVP без LLM)")
    demo.add_argument("--docs", type=str, default="docs/sample_docs", help="Папка с документами (.txt/.pdf)")
    demo.add_argument("--query", type=str, required=True, help="Вопрос пользователя")
    demo.add_argument("--top_k", type=int, default=5, help="Сколько чанков вернуть")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "rag_demo":
        docs_path = Path(args.docs)
        docs = load_documents_from_folder(docs_path)
        if not docs:
            raise SystemExit(f"Не найдено документов в {docs_path} (ожидаются .txt/.pdf)")

        rag = RagPipeline()
        rag.build_index(docs)
        result = rag.ask(args.query, top_k=args.top_k)
        print(result.as_text())


if __name__ == "__main__":
    main()
