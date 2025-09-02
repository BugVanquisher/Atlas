from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    j = await req.json()
    max_tokens = j.get("max_tokens", 50)
    pt = 20
    ct = min(max_tokens, 30)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "(mock) hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
        "model": j.get("model", "mock-model"),
    }
