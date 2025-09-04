#!/usr/bin/env python3
"""
Simple async load tester for Atlas Gateway.

Usage:
  python examples/load_test.py --concurrency 50 --requests 500 \
      --url http://localhost:8080/v1/chat/completions \
      --api-key test-key-123
"""

import argparse
import asyncio
import random
import time

import httpx


async def worker(idx: int, url: str, api_key: str, results: list):
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "mock",
                    "messages": [{"role": "user", "content": f"hello {idx}"}],
                    "max_tokens": 5,
                },
            )
            results.append(r.status_code)
        except Exception as e:
            print(f"[{idx}] error: {e}")
            results.append(0)


async def run_load(url: str, api_key: str, concurrency: int, total: int):
    results = []
    sem = asyncio.Semaphore(concurrency)

    async def bounded_worker(i):
        async with sem:
            await worker(i, url, api_key, results)

    start = time.perf_counter()
    await asyncio.gather(*(bounded_worker(i) for i in range(total)))
    dur = time.perf_counter() - start

    # summary
    print(f"\nCompleted {len(results)} requests in {dur:.2f}s")
    for code in sorted(set(results)):
        if code == 0:
            print("Errors:", results.count(code))
        else:
            print(f"HTTP {code}: {results.count(code)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(run_load(args.url, args.api_key, args.concurrency, args.requests))


if __name__ == "__main__":
    main()
