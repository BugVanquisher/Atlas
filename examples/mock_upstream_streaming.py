#!/usr/bin/env python3
"""
Mock upstream server with streaming support for Atlas testing.
Run with:
  uvicorn examples.mock_upstream_streaming:app --reload --port 9000
"""

import asyncio
import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    logger.info(f"Received request: {body}")
    if body.get("stream"):

        async def gen():
            # emit 3 fake delta chunks
            for i in range(3):
                chunk = {
                    "id": "mock",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {"content": f"chunk-{i} "},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.2)
            logger.info("Stream complete")
            # end of stream
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    else:
        # non-stream response with fake usage
        return JSONResponse(
            {
                "id": "mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 42},
            }
        )
