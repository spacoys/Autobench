import re, numpy as np
from bert_score import score as bert_score

def normalize(text): return re.sub(r'\s+', ' ', text.lower().strip())

def exact_match(pred, true): return normalize(pred) == normalize(true)

def f1_score(pred, true):
    pred_tokens = set(normalize(pred).split())
    true_tokens = set(normalize(true).split())
    if not true_tokens: return 0.0
    inter = pred_tokens & true_tokens
    if not inter: return 0.0
    prec = len(inter) / len(pred_tokens) if pred_tokens else 0.0
    rec = len(inter) / len(true_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def numeric_match(pred: str, true: str, tol=1e-3) -> bool:
    import re

    # Приводим tol к float на случай, если он пришёл строкой из YAML
    try:
        tol = float(tol)
    except (TypeError, ValueError):
        tol = 1e-3

    def extract_numbers(text: str):
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)
        pattern = r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?'
        numbers = []
        for f in re.findall(pattern, text):
            try:
                numbers.append(float(f))
            except ValueError:
                continue
        return numbers

    # Приводим эталон к float
    try:
        true_val = float(str(true).replace(',', '.').strip())
    except (TypeError, ValueError):
        return str(pred).strip() == str(true).strip()

    # Сравниваем с каждым числом из ответа
    for num in extract_numbers(pred):
        if abs(num - true_val) <= tol:
            return True

    return False

def evaluate_responses(results, config):
    metrics = config.get('metrics', ['exact_match', 'f1'])
    numeric_tol = config.get('numeric_tolerance', 1e-5)
    detailed = []

    for item in results:
        row = {'predicted': item['predicted'], 'true': item['true_answer']}
        for m in metrics:
            if m == 'exact_match':
                row[m] = exact_match(item['predicted'], item['true_answer'])
            elif m == 'f1':
                row[m] = f1_score(item['predicted'], item['true_answer'])
            elif m == 'numeric_match':
                row[m] = numeric_match(item['predicted'], item['true_answer'], numeric_tol)
        detailed.append(row)

    # BERTScore считается пакетно для всех сразу
    if 'bert_score' in metrics:
        from bert_score import score as bs
        preds = [d['predicted'] for d in detailed]
        trues = [d['true'] for d in detailed]
        _, _, f1_vals = bs(preds, trues, lang='ru', verbose=False)
        for d, v in zip(detailed, f1_vals.tolist()):
            d['bert_score'] = v

    summary = {}
    for m in metrics:
        if detailed and m in detailed[0]:
            vals = [d[m] for d in detailed if isinstance(d[m], (bool, float, int))]
            if vals:
                summary[m] = sum(vals) / len(vals) if m in ['exact_match', 'numeric_match'] else float(np.mean(vals))
    return {'summary': summary, 'detailed': detailed}