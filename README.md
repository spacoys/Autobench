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
