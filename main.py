import os
import yaml
import json
import asyncio
import argparse
from dotenv import load_dotenv
from src.runner import run_benchmark
from src.utils import ensure_dir

load_dotenv(override=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def safe_name(s: str) -> str:
    for ch in '/\\:*?"<>|':
        s = s.replace(ch, "-")
    return s


async def main():
    args = parse_args()
    print(f"Используется конфиг: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = await run_benchmark(config)

    out_dir = config["output"].get("results_dir", "results/")
    ensure_dir(out_dir)

    benchmark_name = safe_name(config["benchmark"]["name"])
    model_name = safe_name(config["model"]["name"])
    timestamp = int(asyncio.get_running_loop().time())

    filename = f"{out_dir}/{benchmark_name}_{model_name}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {filename}")
    print("Summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
