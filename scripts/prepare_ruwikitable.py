import pandas as pd
import json
from pathlib import Path

# Читаем JSONL-файл
df = pd.read_json("data/train.jsonl", lines=True)

# Посмотрим на колонки
print("Колонки:", df.columns.tolist())
print("Всего записей:", len(df))

# Приводим к формату AutoBench
data = []
for _, row in df.iterrows():
    entry = {
        "question": str(row["question"]).strip(),
        "answer": str(row["answer"]).strip(),
        # Поле table — это CSV-строка. Можно передать как есть или преобразовать в Markdown
        "document": str(row["table"])[:800].strip(),
        # Метаданные (опционально)
        "_type": row.get("question_type"),
        "_reasoning": row.get("reasoning"),
    }
    data.append(entry)

# Сохраняем
Path("data").mkdir(exist_ok=True)
with open("data/ruwikitable.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Сохранено {len(data)} записей в data/ruwikitable.json")
print(f"Пример вопроса: {data[0]['question'][:150]}")
print(f"Пример ответа: {data[0]['answer'][:100]}")
print(f"Длина таблицы: {len(data[0]['document'])} символов")