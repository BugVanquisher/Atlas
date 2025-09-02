# Atlas: Traffic Governance & Capacity Planner for LLM Inference

Atlas is a **traffic management and forecasting layer** designed to sit in front of 
[vLLM](https://github.com/vllm-project/vllm) or any OpenAI-compatible inference server.


## Why Atlas?
Governance layer on top of vLLM

## Features (Phase 1):
•	Quota enforcement (daily/monthly)

•	Rate limiting (QPS + burst)

•	Usage endpoint (/v1/usage)

•	Admin endpoint (/v1/admin/keys)

•	Proxy to vLLM (OpenAI-compatible)

•	Metrics (/metrics)

## Quickstart

```
cp .env.example .env
docker compose -f infra/docker-compose.yml up --build
bash examples/curl_usage.sh
```