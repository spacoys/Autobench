from datasets import load_dataset
import json
from pathlib import Path

Path("data").mkdir(exist_ok=True)

def save(name, data):
    with open(f"data/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ {name}: {len(data)} записей")

# RAG (русские поля)
for cfg in ["cuad", "finqa", "expertqa"]:
    ds = load_dataset("CMCenjoyer/ragbench-ru", cfg, split="test")
    data = [{"question": r["question_ru"], "answer": r["response_ru"], 
             "document": str(r["documents_sentences_ru"])[:5000]} for r in ds]
    save(f"ragbench_{cfg}", data)

# TableQA (пример на основе 2Columns1Row)
try:
    ds = load_dataset("kurumikz/Question-answeringsmall-ru", split="test[:100]")
    data = [{"question": r["question"], "answer": r["answer"], "document": r.get("context","")} for r in ds]
    save("tableqa_sample", data)
except Exception as e:
    print(f"TableQA: {e}")

from datasets import load_dataset

# Загружаем датасет, указав путь к файлу напрямую
ds = load_dataset(
    "parquet",
    data_files="hf://datasets/t-tech/T-math@main/data/train-00000-of-00001.parquet",
    split="train"
)

# Далее сохраняем в JSON
import json
from pathlib import Path

Path("data").mkdir(exist_ok=True)
data = [{"question": r["question"], "answer": str(r["verifiable_answer"]), "document": ""} for r in ds]

with open("data/calc_tmath.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ calc_tmath: {len(data)} записей")

ds = load_dataset("RusNLPWorld/RuFinQA", split="test[:200]")
data = [{"question": r["question"], "answer": str(r["answer"]), "document": r.get("context","")} for r in ds]
save("calc_finqa", data)