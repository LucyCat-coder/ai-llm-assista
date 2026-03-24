# Дипломный проект: RAG-шаблон “из коробки”

Этот репозиторий — стартовая структура диплома под **RAG (Retrieval-Augmented Generation)**:
- чанкинг текста
- “эмбеддинги” (TF‑IDF) и cosine similarity
- простой retrieval (top‑k чанков)
- CLI-демо

> Важно: TF‑IDF используется как лёгкая “векторизация” для MVP.
> На следующих этапах можно заменить на sentence-transformers и векторную БД.

## Быстрый старт (локально)
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск RAG демо
```bash
python -m src.cli.app rag_demo --query "Нужен ли фундамент?"
python -m src.cli.app rag_demo --docs docs/sample_docs --query "Что такое RAG?"
```

## Где лежат документы
- Примеры: `docs/sample_docs/`
- Свои документы складывайте в `data/` (и НЕ коммитьте большие/закрытые данные)

## Структура
- `src/core/chunking.py` — разбиение на чанки (chunk + overlap)
- `src/core/retrieval.py` — TF‑IDF + cosine similarity
- `src/core/rag_pipeline.py` — пайплайн RAG
- `src/parsers/loaders.py` — загрузка txt/pdf
- `src/cli/app.py` — CLI

## Следующие шаги диплома
1) Добавить тест-набор вопросов (Q/A) и метрики retrieval (precision@k)
2) Подключить LLM и генерировать ответ по top‑chunks
3) Сделать Web-интерфейс (Streamlit/FastAPI) и деплой
