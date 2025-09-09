[![GitHub release](https://img.shields.io/github/v/release/BugVanquisher/Atlas)](https://github.com/BugVanquisher/Atlas/releases) ![Build](https://github.com/BugVanquisher/Atlas/actions/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/gh/BugVanquisher/Atlas/branch/main/graph/badge.svg)](https://codecov.io/gh/BugVanquisher/Atlas) ![License](https://img.shields.io/github/license/BugVanquisher/Atlas) 

# Atlas Gateway

**Atlas** is an intelligent traffic governance and quota management gateway for LLM inference (vLLM, OpenAI API compatible).  
It enforces **rate limits, daily/monthly quotas, advanced streaming support, and intelligent observability** as a production-ready layer in front of your model server.

---

## ✨ Features (Phase 2 Complete - v0.2.5)

### 🛡️ **Core Gateway Features**
- ✅ Health check with component status (`/healthz`)
- ✅ Admin API to register API keys and limits (`/v1/admin/keys`)
- ✅ Quota enforcement (daily & monthly with intelligent refunding)
- ✅ Rate limiting (QPS + burst with token bucket algorithm)
- ✅ Self-serve usage endpoint (`/v1/usage`)
- ✅ Proxy to upstream (OpenAI-style `/v1/chat/completions`)
- ✅ Token accounting from `usage.total_tokens`
- ✅ Prometheus metrics at `/metrics`

### 🔄 **Enhanced Streaming Support (New in v0.2.0)**
- ✅ **Robust streaming architecture** with pre-flight validation
- ✅ **Intelligent token management** with dynamic reservations
- ✅ **Priority-based resource allocation** (low/normal/high/critical)
- ✅ **Smart quota refunding** based on actual usage
- ✅ **Enhanced error handling** with graceful degradation
- ✅ **Real-time usage tracking** with multiple parsing strategies
- ✅ **Streaming-specific metrics** and observability

### ⭐ **Priority & Resource Management**
- ✅ **Request-level priority** override via `x-request-priority` header
- ✅ **Dynamic token reservations** based on priority levels
- ✅ **Priority-aware quota management** and enforcement
- ✅ **Resource allocation optimization** for mixed workloads

### 🔮 **Traffic Forecasting & Analytics (New in v0.2.5)**
- ✅ **Usage pattern analysis** with peak/trend detection
- ✅ **Capacity planning recommendations** with growth projections
- ✅ **Forecasting API endpoints** (`/v1/forecasting/*`)
- ✅ **Historical metrics collection** with Redis storage
- ✅ **System-wide analytics** and admin insights
- ✅ **Cost tracking and optimization** recommendations

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
```bash
git clone https://github.com/BugVanquisher/Atlas.git
cd Atlas
cp .env.example .env

# Start with Docker (includes Redis, mock upstream, and gateway)
docker compose -f infra/docker-compose.yml up --build
```

### Option 2: Local Development
```bash
git clone https://github.com/BugVanquisher/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Redis locally
redis-server

# Start Atlas Gateway
uvicorn gateway.main:app --reload --port 8080
```

### Try it out
```bash
# Basic usage example
bash examples/curl_usage.sh

# Test enhanced streaming features
python examples/test_streaming.py --api-key test-key-123 --test all

# Explore forecasting capabilities
python examples/forecasting_demo.py demo --api-key test-forecast
```

---

## 📋 API Reference

### **Core Gateway Endpoints**
- `GET /healthz` - Health check with component status
- `POST /v1/admin/keys` - Create/update API keys and limits
- `GET /v1/usage` - Get current usage and limits
- `POST /v1/chat/completions` - Proxy to upstream with governance
- `GET /metrics` - Prometheus metrics

### **Forecasting & Analytics (New)**
- `GET /v1/forecasting/forecast` - Get traffic predictions
- `GET /v1/forecasting/usage-patterns` - Analyze historical patterns  
- `GET /v1/forecasting/capacity-recommendations` - Get optimization suggestions
- `GET /v1/forecasting/admin/system-metrics` - System-wide analytics

### **Enhanced Request Headers**
```bash
# Set request priority (overrides API key default)
x-request-priority: high

# Standard authentication
Authorization: Bearer your-api-key
```

---

## 🎛️ Configuration

### **Environment Variables**
```env
# Core settings
UPSTREAM_BASE_URL=http://your-llm-server:9000
REDIS_URL=redis://localhost:6379/0
ADMIN_API_KEY=your-secure-admin-key

# Quota defaults
DEFAULT_DAILY_LIMIT=100000
DEFAULT_MONTHLY_LIMIT=1000000
DEFAULT_RATE_PER_SEC=2.0
DEFAULT_BURST=5

# Enhanced streaming settings (New in v0.2.0)
DEFAULT_STREAM_RESERVATION=256
MAX_STREAM_RESERVATION=2000
STREAM_CONNECT_TIMEOUT=30
STREAM_READ_TIMEOUT=300
STREAM_DEBUG_LOGGING=false
```

### **Priority-Based Resource Allocation**
```json
{
  "low": "5% of daily limit",
  "normal": "10% of daily limit", 
  "high": "20% of daily limit",
  "critical": "30% of daily limit"
}
```

---

## 🧪 Testing & Examples

### **Comprehensive Test Suite**
```bash
# Core functionality tests
PYTHONPATH=. pytest -q

# Enhanced streaming tests
python examples/test_streaming.py --api-key test-key --test all

# Forecasting system tests  
python examples/forecasting_demo.py --help
```

### **Performance Benchmarking**
```bash
# Load testing
python examples/load_test.py --api-key test-key --concurrency 50 --requests 500

# Streaming performance test
python examples/test_streaming.py --api-key test-key --test benchmark --concurrent 10
```

### **Interactive Examples**
```bash
# Basic gateway usage
bash examples/curl_usage.sh

# Complete forecasting demo
python examples/forecasting_demo.py demo --api-key test-forecast

# Get capacity recommendations
python examples/forecasting_demo.py recommendations --api-key test-key
```

---

## 📊 Observability & Monitoring

### **Prometheus Metrics**
- `atlas_requests_total` - Total requests by API key, priority, and route
- `atlas_tokens_used_total` - Token consumption by API key and priority
- `atlas_quota_rejections_total` - Quota violations by scope (daily/monthly)
- `atlas_rate_limit_rejections_total` - Rate limit violations
- `atlas_streaming_requests_total` - Streaming requests by priority
- `atlas_streaming_duration_seconds` - Stream duration distribution
- `atlas_token_refunds_total` - Token refunds by reason

### **Health Monitoring**
```bash
curl http://localhost:8080/healthz
```
```json
{
  "ok": true,
  "status": "healthy",
  "components": {
    "redis": "healthy",
    "forecasting": "healthy"
  },
  "timestamp": "2025-09-08T10:00:00Z"
}
```

### **Usage Analytics**
```bash
# Get your usage patterns
curl -H "Authorization: Bearer your-key" \
  "http://localhost:8080/v1/forecasting/usage-patterns?days_back=30"

# Get capacity recommendations
curl -H "Authorization: Bearer your-key" \
  "http://localhost:8080/v1/forecasting/capacity-recommendations"
```

---

## 🏗️ Architecture

### **Component Overview**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │───▶│  Atlas Gateway  │───▶│  LLM Upstream   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │ Redis (Metrics) │
                       └─────────────────┘
```

### **Key Components**
- **Gateway Core**: Request routing, auth, rate limiting
- **Streaming Handler**: Enhanced streaming with intelligent token management  
- **Quota Manager**: Usage tracking and enforcement with refunding
- **Forecasting Engine**: Pattern analysis and capacity planning
- **Metrics Collector**: Historical data aggregation and storage

---

## 🛣️ Roadmap

### **✅ Phase 1 (v0.1.0): Foundation**
- Core quota/rate limiting
- Basic streaming support
- Prometheus metrics
- Admin APIs

### **✅ Phase 1.5 (v0.1.5): Request Criticality** 
- Priority-based resource allocation
- Enhanced admin operations

### **✅ Phase 2 (v0.2.0): Advanced Streaming & Analytics**
- **Robust streaming architecture** with pre-flight validation
- **Intelligent token management** and dynamic reservations
- **Enhanced observability** and metrics

### **✅ Phase 2.5 (v0.2.5): Traffic forecasting**
- **Traffic forecasting** and capacity planning

### **🚧 Phase 3 (v0.3.0): Intelligence & Optimization** 
- Multi-provider intelligent routing
- Cost-aware request distribution  
- Advanced admin dashboard
- Real-time anomaly detection
- Automated capacity scaling

### **🔮 Phase 4 (v0.4.0): Enterprise Features**
- Multi-tenant dashboards
- Advanced billing and cost allocation
- SLA monitoring and enforcement
- Compliance and audit trails

---

## 🎯 Use Cases

### **Production LLM Deployments**
- **API Gateway**: Centralized governance for multiple LLM services
- **Cost Control**: Prevent runaway usage with intelligent quotas
- **Performance Optimization**: Priority-based resource allocation
- **Capacity Planning**: Data-driven scaling decisions

### **Multi-Tenant SaaS Platforms**
- **Per-customer quotas**: Isolated usage tracking and enforcement
- **Fair resource sharing**: Priority-based allocation across tenants
- **Usage analytics**: Customer insights and billing optimization
- **Predictive scaling**: Forecast-driven infrastructure planning

### **Enterprise AI Operations**
- **Governance**: Centralized policy enforcement
- **Observability**: Comprehensive monitoring and alerting
- **Cost optimization**: Usage pattern analysis and recommendations
- **Compliance**: Audit trails and usage reporting

---

## 🤝 Contributing

We welcome contributions! 🚀

### **Development Setup**
```bash
git clone https://github.com/BugVanquisher/Atlas.git
cd Atlas
python3 -m venv .venv
source .venv/bin/activate
pip install -r dev-requirements.txt
```

### **Common Commands**
```bash
make lint        # run black, isort, flake8
make format      # auto-format code  
make test        # run pytest (unit tests only)
make run         # start gateway on port 8080
```

### **Areas for Contribution**
- **Phase 3 Features**: Multi-provider routing, advanced dashboards
- **Performance**: Optimization and scaling improvements
- **Documentation**: Guides, tutorials, and API documentation
- **Testing**: Additional test coverage and integration tests

---

## 📞 Support & Community

### **Getting Help**
- 🐛 **Issues**: [GitHub Issues](https://github.com/BugVanquisher/Atlas/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/BugVanquisher/Atlas/discussions)
- 📚 **Documentation**: [Atlas Docs](https://github.com/BugVanquisher/Atlas/docs)

### **Reporting Issues**
When reporting issues, please include:
- Atlas Gateway version (`/healthz` endpoint output)
- Configuration details (sanitized)
- Request details and error logs
- Expected vs actual behavior

---

## 📄 License

**Apache 2.0 License** - see [LICENSE](LICENSE) file for details.

---

## 🏷️ Version History

- **v0.2.5** (2025-09-08): Traffic forecasting, capacity planning, advanced analytics
- **v0.2.0** (2025-08-20): Enhanced streaming, intelligent token management
- **v0.1.5** (2025-08-15): Request criticality and priority support  
- **v0.1.0** (2025-07-01): Initial release with core gateway features

**Full Changelog**: [Releases](https://github.com/BugVanquisher/Atlas/releases)