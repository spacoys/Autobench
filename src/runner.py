import asyncio, time
from tqdm import tqdm
from .benchmark_loader import load_benchmark
from .model_client import ModelClient
from .evaluator import evaluate_responses

async def run_benchmark(config):
    dataset = load_benchmark(config)
    print(f"Loaded {len(dataset)} examples.")
    client = ModelClient(config['model'])
    use_context = config['model'].get('use_context', True)

    # Асинхронный прогон
    async def ask_all():
        tasks = [client.ask(item['question'], item.get('context') if use_context else None) for item in dataset]
        return await asyncio.gather(*tasks)

    start = time.time()
    predictions = await ask_all()
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f} seconds.")

    results = []
    for item, pred in zip(dataset, predictions):
        results.append({
            'question': item['question'],
            'true_answer': item['answer'],
            'predicted': pred
        })

    eval_res = evaluate_responses(results, config.get('evaluation', {}))
    eval_res['metadata'] = {
        'num_examples': len(dataset),
        'model': config['model']['name'],
        'benchmark': config['benchmark']['name'],
        'elapsed_seconds': elapsed
    }
    return eval_res