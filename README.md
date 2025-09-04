[![GitHub release](https://img.shields.io/github/v/release/BugVanquisher/Atlas)](https://github.com/BugVanquisher/Atlas/releases)

# Atlas

**Atlas** is a traffic governance and quota management gateway for LLM inference (vLLM, OpenAI API compatible).  
It enforces **rate limits, daily/monthly quotas, and observability** as a thin layer in front of your model server.

---

## ✨ Features (Phase 1)
- ✅ Health check (`/healthz`)
- ✅ Admin API to register API keys and limits (`/v1/admin/keys`)
- ✅ Quota enforcement (daily & monthly)
- ✅ Rate limiting (QPS + burst)
- ✅ Self-serve usage endpoint (`/v1/usage`)
- ✅ Proxy to upstream (OpenAI-style `/v1/chat/completions`)
- ✅ Token accounting from `usage.total_tokens`
- ✅ Prometheus metrics at `/metrics`

---

## 🚀 Quickstart

```bash
git clone https://github.com/BugVanquisher/Atlas.git
cd Atlas
cp .env.example .env

# run locally with docker
docker compose -f infra/docker-compose.yml up --build
```

Try it out:
```
bash examples/curl_usage.sh
```

---

## 🧪 Tests
•	Unit tests use fakeredis (no external dependencies).

•	Run all tests:
```
PYTHONPATH=. pytest -q
```

---

## 📊 Observability

Prometheus metrics exposed at /metrics:

•	atlas_requests_total (by API key)

•	atlas_tokens_used_total (by API key)

•	atlas_quota_rejections_total

---

## 📍 Roadmap
•	Phase 1 (v0.1.0): Core quota/rate limiting ✅

•	Phase 1.5: Add request criticality support ✅

•	Phase 2: Streaming support, pre-reservations, advanced admin ops (WIP)

•	Phase 3: Traffic forecasting, capacity planning

•	Phase 4: Dashboards, billing, multi-tenant observability

---

## 📄 License

Apache 2.0

---


# Contributing to Atlas

We welcome contributions! 🚀

## Dev Setup

```bash
git clone https://github.com/BugVanquisher/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -r dev-requirements.txt
```
## Common Commands
```
make lint        # run black, isort, flake8
make format      # auto-format code
make test        # run pytest (unit tests only)
make run         # start gateway on port 8080
```