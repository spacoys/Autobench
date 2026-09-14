import json, csv, pandas as pd
from pathlib import Path
from typing import List, Dict

def load_benchmark(config: Dict) -> List[Dict]:
    path = Path(config['benchmark']['path'])
    fmt = config['benchmark'].get('format', 'json')
    q_col = config['benchmark']['question_column']
    a_col = config['benchmark']['answer_column']
    c_col = config['benchmark'].get('context_column')

    if fmt == 'json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif fmt == 'csv':
        data = pd.read_csv(path).to_dict(orient='records')
    elif fmt == 'parquet':
        data = pd.read_parquet(path).to_dict(orient='records')
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    processed = []
    for item in data:
        entry = {'question': str(item[q_col]), 'answer': str(item[a_col])}
        if c_col and c_col in item:
            entry['context'] = str(item[c_col])
        processed.append(entry)
    return processed