from pathlib import Path

Path("configs").mkdir(exist_ok=True)

TEMPLATE = """benchmark:
  name: "{name}"
  path: "data/{name}.json"
  format: "json"
  has_context: true
  question_column: "question"
  answer_column: "answer"
  context_column: "document"

model:
  name: "openai/gpt-oss-20b"
  temperature: 0.1
  max_tokens: 512
  use_context: true

evaluation:
  metrics: ["exact_match", "f1"]
  numeric_tolerance: 1e-5

output:
  results_dir: "results/"
  save_detailed: true
"""

names = [
    "ragbench_cuad",
    "ragbench_cuad_sample",
    "ragbench_finqa",
    "ragbench_finqa_sample",
    "ragbench_expertqa",
    "ragbench_expertqa_sample",
]

for full_name in names:
    # имя файла без префикса ragbench_
    short = full_name.replace("ragbench_", "")
    path = Path(f"configs/{short}.yaml")
    path.write_text(TEMPLATE.format(name=full_name), encoding="utf-8")
    print(f"✅ {path}  ({path.stat().st_size} байт)")