#!/usr/bin/env python3
"""
Enhanced streaming example and test script for Atlas Gateway.

Usage:
    python examples/test_streaming.py --api-key test-key-123 --url http://localhost:8080
"""

import argparse
import asyncio
import json
import time

import httpx


async def test_streaming_request(
    url: str,
    api_key: str,
    max_tokens: int = 100,
    priority: str = "normal",
    message: str = "Write a short story about a robot",
):
    """Test a single streaming request"""

    print("\n🚀 Testing streaming request:")
    print(f"   Priority: {priority}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Message: {message[:50]}...")

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-request-priority": priority,
                },
                json={
                    "model": "mock",
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                print(f"   Status: {response.status_code}")

                if response.status_code != 200:
                    error_content = await response.aread()
                    print(f"   Error: {error_content}")
                    return False

                chunks_received = 0
                total_content = ""

                async for chunk in response.aiter_bytes():
                    if chunk:
                        chunks_received += 1
                        chunk_str = chunk.decode("utf-8")

                        # Parse streaming chunks
                        for line in chunk_str.strip().split("\n"):
                            if line.startswith("data: "):
                                data_content = line[6:].strip()
                                if data_content == "[DONE]":
                                    continue

                                try:
                                    chunk_json = json.loads(data_content)
                                    if "choices" in chunk_json:
                                        delta = chunk_json["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            total_content += content
                                            print(
                                                f"   Chunk {chunks_received}: {content}",
                                                end="",
                                                flush=True,
                                            )

                                    # Check for usage data
                                    if "usage" in chunk_json:
                                        usage = chunk_json["usage"]
                                        print(f"\n   Usage: {usage}")

                                except json.JSONDecodeError:
                                    pass

                duration = time.time() - start_time
                print(f"\n   ✅ Completed in {duration:.2f}s")
                print(f"   📊 Chunks received: {chunks_received}")
                print(f"   📝 Total content length: {len(total_content)}")
                return True

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n   ❌ Failed after {duration:.2f}s: {e}")
        return False


async def test_quota_limits(url: str, api_key: str):
    """Test quota enforcement with streaming"""

    print("\n🔒 Testing quota limits...")

    # First, get current usage
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{url}/v1/usage", headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            usage = response.json()
            print(f"   Current usage: {usage['daily_used']}/{usage['daily_limit']} daily")
            print(f"   Current usage: {usage['monthly_used']}/{usage['monthly_limit']} monthly")

            # Try to exceed daily limit
            remaining = usage["daily_limit"] - usage["daily_used"]
            if remaining > 0:
                print(
                    f"   Attempting to exceed daily limit (requesting {remaining + 100} tokens)..."
                )

                success = await test_streaming_request(
                    url,
                    api_key,
                    max_tokens=remaining + 100,
                    message="Generate a very long response",
                )

                if not success:
                    print("   ✅ Quota enforcement working correctly")
                else:
                    print("   ⚠️  Request succeeded when it should have been blocked")
            else:
                print("   Daily limit already reached")
        else:
            print(f"   Failed to get usage: {response.status_code}")


async def test_different_priorities(url: str, api_key: str):
    """Test requests with different priorities"""

    print("\n⭐ Testing priority levels...")

    priorities = ["low", "normal", "high", "critical"]

    for priority in priorities:
        success = await test_streaming_request(
            url,
            api_key,
            max_tokens=50,
            priority=priority,
            message=f"Test message for {priority} priority",
        )

        if success:
            print(f"   ✅ {priority.upper()} priority: Success")
        else:
            print(f"   ❌ {priority.upper()} priority: Failed")

        await asyncio.sleep(0.5)  # Brief delay between requests


async def test_error_scenarios(url: str, api_key: str):
    """Test error handling scenarios"""

    print("\n🔥 Testing error scenarios...")

    # Test with invalid API key
    print("   Testing invalid API key...")
    await test_streaming_request(url, "invalid-key", max_tokens=50, message="This should fail")

    # Test with malformed request
    print("   Testing malformed request...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "mock",
                    "messages": "invalid-format",  # Should be array
                    "stream": True,
                },
            )
            print(f"   Malformed request status: {response.status_code}")
    except Exception as e:
        print(f"   Malformed request error: {e}")


async def benchmark_streaming(url: str, api_key: str, concurrent_requests: int = 5):
    """Benchmark streaming performance"""

    print(f"\n📈 Benchmarking with {concurrent_requests} concurrent streams...")

    start_time = time.time()

    # Create concurrent streaming requests
    tasks = []
    for i in range(concurrent_requests):
        task = test_streaming_request(
            url, api_key, max_tokens=30, message=f"Concurrent test request {i + 1}"
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time
    successes = sum(1 for r in results if r is True)

    print(f"   📊 Results: {successes}/{concurrent_requests} successful")
    print(f"   ⏱️  Total time: {duration:.2f}s")
    print(f"   🚀 Requests per second: {concurrent_requests / duration:.2f}")


async def main():
    parser = argparse.ArgumentParser(description="Test Atlas Gateway streaming functionality")
    parser.add_argument("--url", default="http://localhost:8080", help="Gateway URL")
    parser.add_argument("--api-key", required=True, help="API key for testing")
    parser.add_argument(
        "--test",
        choices=["basic", "quota", "priority", "error", "benchmark", "all"],
        default="all",
        help="Which tests to run",
    )
    parser.add_argument(
        "--concurrent", type=int, default=5, help="Concurrent requests for benchmark"
    )

    args = parser.parse_args()

    print("🧪 Atlas Gateway Streaming Test Suite")
    print(f"URL: {args.url}")
    print(f"API Key: {args.api_key[:8]}...")

    if args.test in ["basic", "all"]:
        await test_streaming_request(args.url, args.api_key)

    if args.test in ["quota", "all"]:
        await test_quota_limits(args.url, args.api_key)

    if args.test in ["priority", "all"]:
        await test_different_priorities(args.url, args.api_key)

    if args.test in ["error", "all"]:
        await test_error_scenarios(args.url, args.api_key)

    if args.test in ["benchmark", "all"]:
        await benchmark_streaming(args.url, args.api_key, args.concurrent)

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
