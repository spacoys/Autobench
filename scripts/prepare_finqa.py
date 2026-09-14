import json
from pathlib import Path
from datasets import load_dataset

# Лимит символов на контекст (для Groq free tier)
MAX_CONTEXT_CHARS = 400

# Сколько примеров брать (для быстрого теста — 200; для полного — None)
LIMIT = 200


def flatten_dict(d, indent=0) -> str:
    """Разворачивает словарь context в читаемый текст."""
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, (int, float, bool)):
        return str(d)
    if isinstance(d, list):
        return "\n".join(flatten_dict(x, indent) for x in d)
    if isinstance(d, dict):
        lines = []
        for k, v in d.items():
            if v is None:
                continue
            prefix = "  " * indent
            if isinstance(v, dict):
                lines.append(f"{prefix}{k}:")
                lines.append(flatten_dict(v, indent + 1))
            elif isinstance(v, list):
                lines.append(f"{prefix}{k}:")
                lines.append(flatten_dict(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")
        return "\n".join(lines)
    return str(d)


def make_context(row) -> str:
    """Собирает контекст из полей context + reasoning_steps."""
    parts = []

    # 1. Финансовые данные из context
    ctx = row.get("context")
    if ctx:
        parts.append("[Финансовые данные]")
        parts.append(flatten_dict(ctx))

    # 2. Шаги рассуждения (опционально — дают модели подсказку, что искать)
    steps = row.get("reasoning_steps")
    if steps:
        parts.append("\n[План решения]")
        for s in steps:
            desc = s.get("description", "")
            if desc:
                parts.append(f"  Шаг {s.get('step', '?')}: {desc}")

    return "\n".join(parts)


# ---------------- Загрузка ----------------
print("Загружаю RuFinQA...")
split = f"train[:{LIMIT}]" if LIMIT else "train"
ds = load_dataset("RusNLPWorld/RuFinQA", split=split)
print(f"Загружено: {len(ds)}")

# ---------------- Конвертация ----------------
data = []
skipped = 0
for row in ds:
    question = str(row.get("question", "")).strip()

    # Эталонный ответ — числовой (для numeric_match)
    numeric_answer = row.get("numeric_answer")
    final_answer = row.get("final_answer", "")

    # Если numeric_answer нет, пробуем вытащить число из final_answer
    answer = str(numeric_answer) if numeric_answer is not None else str(final_answer)

    if not question or not answer:
        skipped += 1
        continue

    context = make_context(row)[:MAX_CONTEXT_CHARS]

    entry = {
        "question": question,
        "answer": answer,
        "document": context,
        # метаданные для анализа
        "_id": row.get("id"),
        "_type": row.get("type"),
        "_skill": row.get("skill"),
        "_difficulty": row.get("difficulty"),
        "_tolerance": row.get("tolerance"),
        "_final_answer": final_answer,
    }
    data.append(entry)

# ---------------- Сохранение ----------------
Path("data").mkdir(exist_ok=True)
out = Path("data/calc_finqa.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------- Отчёт ----------------
print(f"\n✅ Сохранено: {len(data)} (пропущено: {skipped})")
print(f"📁 Файл: {out}")

ctx_lens = [len(e["document"]) for e in data]
ans_lens = [len(e["answer"]) for e in data]
print(f"📏 Контекст: min={min(ctx_lens)}, max={max(ctx_lens)}, avg={sum(ctx_lens)//len(ctx_lens)}")
print(f"📏 Ответ:    min={min(ans_lens)}, max={max(ans_lens)}, avg={sum(ans_lens)//len(ans_lens)}")

# Типы задач
from collections import Counter
types = Counter(e["_type"] for e in data)
print(f"\n📊 Типы задач:")
for t, c in types.most_common():
    print(f"   {t}: {c}")

print(f"\n🔍 Пример 1:")
print(f"   Вопрос: {data[0]['question'][:150]}")
print(f"   Ответ:  {data[0]['answer']}")
print(f"   Final:  {data[0]['_final_answer'][:100]}")