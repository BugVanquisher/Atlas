#!/usr/bin/env bash
set -euo pipefail

ADMIN_KEY=${ADMIN_KEY:-my-admin-key}
API_KEY=${API_KEY:-test-key-123}

curl -s -H "x-admin-key: ${ADMIN_KEY}" -H "content-type: application/json"   -d "{\"api_key\":\"${API_KEY}\",\"daily_limit\":200000,\"monthly_limit\":2000000,\"rate_per_sec\":5,\"burst\":10}"   http://localhost:8080/v1/admin/keys | jq

curl -s -H "Authorization: Bearer ${API_KEY}" -H "content-type: application/json"   -d '{"model":"mock","messages":[{"role":"user","content":"hi"}],"max_tokens":64,"stream":false}'   http://localhost:8080/v1/chat/completions | jq

curl -s -H "Authorization: Bearer ${API_KEY}" http://localhost:8080/v1/usage | jq
