"""
Подготовка датасетов RAG-Bench Russian (cuad, finqa, expertqa) к формату AutoBench.

Запуск:
    python scripts/prepare_ragbench.py
    python scripts/prepare_ragbench.py --split validation
    python scripts/prepare_ragbench.py --max-chars 4000
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset

# ==================== НАСТРОЙКИ ====================
OUT_DIR = Path("data")
DATASET_NAME = "CMCenjoyer/ragbench-ru"
CONFIGS = ["cuad", "finqa", "expertqa"]

# Максимальная длина контекста в символах.
# Для Groq free tier (8000 TPM) рекомендуется 4000–5000.
# Для OpenAI / локальных моделей можно 10000+.
MAX_CONTEXT_CHARS = 1500

# Минимальная длина эталонного ответа, ниже которой запись отбрасывается.
MIN_ANSWER_CHARS = 5
# ===================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="test",
        help="Какой сплит использовать: test / validation / train"
    )
    parser.add_argument(
        "--max-chars", type=int, default=MAX_CONTEXT_CHARS,
        help="Максимальная длина контекста в символах"
    )
    return parser.parse_args()


def flatten_documents(documents_sentences) -> str:
    """
    Превращает вложенный список предложений в один текстовый блок.

    Вход:  [[['0a', 'текст'], ['0b', 'текст']], [['1a', 'текст']]]
    Выход: '[Doc 0]\\nтекст текст\\n\\n[Doc 1]\\nтекст'
    """
    if not documents_sentences:
        return ""

    parts = []
    for doc_idx, doc in enumerate(documents_sentences):
        sentences = []
        for item in doc:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sentences.append(str(item[1]))
            else:
                sentences.append(str(item))
        parts.append(f"[Doc {doc_idx}]\n" + " ".join(sentences))

    return "\n\n".join(parts)


def build_entry(row: dict, max_chars: int) -> dict:
    """Собирает запись из русских полей, если есть; иначе — из английских."""
    question = row.get("question_ru") or row.get("question") or ""
    answer = row.get("response_ru") or row.get("response") or ""
    context = flatten_documents(
        row.get("documents_sentences_ru") or row.get("documents_sentences")
    )

    return {
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "document": context[:max_chars].strip(),
        # метаданные — пригодятся для анализа
        "id": row.get("id"),
        "dataset_name": row.get("dataset_name"),
        "adherence_score": row.get("adherence_score"),
        "relevance_score": row.get("relevance_score"),
        "completeness_score": row.get("completeness_score"),
    }


def is_valid(entry: dict) -> bool:
    """Отбрасывает записи с пустым вопросом/ответом или слишком коротким ответом."""
    return (
        bool(entry["question"])
        and bool(entry["answer"])
        and len(entry["answer"]) >= MIN_ANSWER_CHARS
    )


def process_config(cfg: str, split_name: str, max_chars: int) -> None:
    print(f"\n{'='*60}")
    print(f"Конфиг: {cfg}")
    print(f"{'='*60}")

    ds = load_dataset(DATASET_NAME, cfg)

    if split_name not in ds:
        actual = list(ds.keys())[0]
        print(f"  ⚠️  Сплит '{split_name}' не найден, используем '{actual}'")
        split_name_used = actual
    else:
        split_name_used = split_name

    split = ds[split_name_used]
    print(f"  Сплит: {split_name_used}, всего примеров: {len(split)}")

    data = []
    skipped = 0
    for row in split:
        entry = build_entry(row, max_chars)
        if is_valid(entry):
            data.append(entry)
        else:
            skipped += 1

    if not data:
        print(f"  ❌ Нет валидных записей для {cfg}")
        return

    out_path = OUT_DIR / f"ragbench_{cfg}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Статистика по длинам
    doc_lengths = [len(x["document"]) for x in data]
    ans_lengths = [len(x["answer"]) for x in data]

    print(f"  ✅ Сохранено: {len(data)} (пропущено: {skipped})")
    print(f"  📁 Файл: {out_path}")
    print(f"  📏 Контекст: min={min(doc_lengths)}, "
          f"max={max(doc_lengths)}, "
          f"avg={sum(doc_lengths)//len(doc_lengths)}")
    print(f"  📏 Эталон:   min={min(ans_lengths)}, "
          f"max={max(ans_lengths)}, "
          f"avg={sum(ans_lengths)//len(ans_lengths)}")
    print(f"  🔍 Пример вопроса: {data[0]['question'][:120]}")


def main():
    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Настройки:")
    print(f"  Датасет:      {DATASET_NAME}")
    print(f"  Сплит:        {args.split}")
    print(f"  Max символов: {args.max_chars}")
    print(f"  Конфиги:      {CONFIGS}")

    for cfg in CONFIGS:
        try:
            process_config(cfg, args.split, args.max_chars)
        except Exception as e:
            print(f"  ❌ Ошибка при обработке {cfg}: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print("Готово. Проверьте файлы в data/ragbench_*.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()