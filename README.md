# AutoBench

Инструмент автоматизированного тестирования AI-пайплайнов на открытых русскоязычных бенчмарках.

## Что делает

- Прогоняет датасеты через OpenAI-совместимые модели (Groq, OpenAI, локальные)
- Считает метрики: Exact Match, F1, Numeric Match
- Сохраняет результаты в JSON

## Поддерживаемые бенчмарки

| Тип | Бенчмарк | Источник |
|---|---|---|
| RAG | RAG-Bench Russian | CMCenjoyer/ragbench-ru |
| Расчёты | RuFinQA | RusNLPWorld/RuFinQA |
| Таблицы | RuWikiTable-RAG | kruvcraft/RuWikiTable-RAG |
## Подготовка датасетов

Все датасеты скачиваются и конвертируются в единый формат автоматически.

### RAG-Bench Russian (cuad, finqa, expertqa)

```bash
python scripts/prepare_ragbench.py
```

Результат: `data/ragbench_cuad.json`, `data/ragbench_finqa.json`, `data/ragbench_expertqa.json`.

### RuFinQA (расчётные задачи)

```bash
python scripts/prepare_finqa.py
```

Результат: `data/calc_finqa.json` (200 примеров).

### RuWikiTable-RAG (табличные задачи)

**Шаг 1.** Скачайте JSONL-файл:

например, RuWikiTable-RAG

**Шаг 2.** Конвертируйте:

```bash
python scripts/prepare_ruwikitable.py
```

Результат: `data/tableqa_ruwiki.json`.

---

## Запуск бенчмарков

Каждый бенчмарк запускается одной командой с указанием конфига.

### RAG на юридических документах (cuad)

```bash
python main.py --config configs/ragbench_cuad.yaml
```

### RAG на экспертных вопросах (expertqa)

```bash
python main.py --config configs/ragbench_expertqa.yaml
```

### Расчётные задачи (RuFinQA)

```bash
python main.py --config configs/calc_finqa.yaml
```

### Табличные задачи (RuWikiTable-RAG)

```bash
python main.py --config configs/tableqa_ruwiki.yaml
```

**Пример вывода:**

```
Используется конфиг: configs/calc_finqa.yaml
Loaded 200 examples.
Finished in 245.30 seconds.
Results saved to results/calc_finqa_groq-compound-mini_134208.json
Summary:
  numeric_match: 0.6000
  exact_match: 0.0000
```

**Каждый прогон сохраняется в `results/` отдельным JSON-файлом.**
