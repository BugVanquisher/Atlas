import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


class StreamingError(Exception):
    """Custom exception for streaming-related errors"""

    def __init__(self, message: str, status_code: int = 502):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class TokenUsageParser:
    """Parse token usage from different streaming response formats"""

    @staticmethod
    def parse_openai_chunk(chunk_data: bytes) -> Optional[int]:
        """Parse OpenAI-style streaming chunk for token usage"""
        try:
            chunk_str = chunk_data.decode("utf-8")

            # Handle Server-Sent Events format
            lines = chunk_str.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("data: "):
                    data_content = line[6:].strip()
                    if data_content == "[DONE]":
                        continue

                    try:
                        chunk_json = json.loads(data_content)
                        if isinstance(chunk_json, dict) and "usage" in chunk_json:
                            return int(chunk_json["usage"].get("total_tokens", 0))
                    except json.JSONDecodeError:
                        continue

            # Fallback: try to parse as direct JSON
            try:
                chunk_json = json.loads(chunk_str)
                if isinstance(chunk_json, dict) and "usage" in chunk_json:
                    return int(chunk_json["usage"].get("total_tokens", 0))
            except json.JSONDecodeError:
                pass

        except Exception as e:
            logger.debug(f"Error parsing chunk for usage: {e}")

        return None

    @staticmethod
    def parse_accumulated_response(response_body: bytes) -> int:
        """Parse complete streaming response body for final token count"""
        try:
            response_str = response_body.decode("utf-8")

            # Try multiple parsing strategies

            # Strategy 1: Parse as complete JSON (non-streaming fallback)
            try:
                response_json = json.loads(response_str)
                if isinstance(response_json, dict) and "usage" in response_json:
                    return int(response_json["usage"].get("total_tokens", 0))
            except json.JSONDecodeError:
                pass

            # Strategy 2: Parse streaming format line by line
            max_tokens = 0
            lines = response_str.strip().split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("data: "):
                    data_content = line[6:].strip()
                    if data_content == "[DONE]":
                        continue

                    try:
                        chunk_json = json.loads(data_content)
                        if isinstance(chunk_json, dict) and "usage" in chunk_json:
                            tokens = int(chunk_json["usage"].get("total_tokens", 0))
                            max_tokens = max(max_tokens, tokens)
                    except json.JSONDecodeError:
                        continue

            return max_tokens

        except Exception as e:
            logger.warning(f"Error parsing accumulated response: {e}")
            return 0


class StreamingHandler:
    """Enhanced streaming handler with better error handling and token tracking"""

    def __init__(self, quota_manager, upstream_client, metrics):
        self.quota = quota_manager
        self.upstream = upstream_client
        self.metrics = metrics
        self.parser = TokenUsageParser()

    async def handle_streaming_request(
        self,
        request,
        full_path: str,
        api_key: str,
        limits: Dict[str, Any],
        req_priority: str,
        payload: Dict[str, Any],
        body_bytes: bytes,
    ) -> StreamingResponse:
        """Main entry point for handling streaming requests"""

        # Calculate reservation based on max_tokens
        reservation = self._calculate_reservation(payload, limits)

        # Pre-flight quota check and reservation
        if reservation > 0:
            await self._check_and_reserve_quota(api_key, reservation, limits, req_priority)

        try:
            # Validate upstream connectivity before starting stream
            await self._validate_upstream_connectivity(request, full_path, body_bytes)
        except StreamingError as e:
            # Refund quota on connectivity validation failure
            if reservation > 0:
                await self.quota.refund_tokens(api_key, reservation)
                logger.info(
                    f"Refunded {reservation} tokens due to connectivity validation " f"failure"
                )
            # Convert StreamingError to HTTPException with proper status code
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            # Refund quota on unexpected validation failure
            if reservation > 0:
                await self.quota.refund_tokens(api_key, reservation)
                logger.info(
                    f"Refunded {reservation} tokens due to unexpected validation " f"failure"
                )
            # Convert generic error to 502 Bad Gateway
            raise HTTPException(status_code=502, detail=f"Upstream validation failed: {str(e)}")

        # Return streaming response
        return StreamingResponse(
            self._stream_with_quota_management(
                request, full_path, body_bytes, api_key, req_priority, reservation
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",
            },
        )

    def _calculate_reservation(self, payload: Dict[str, Any], limits: Dict[str, Any]) -> int:
        """Calculate token reservation for streaming request"""
        # Use max_tokens if provided, otherwise use a default based on priority
        max_tokens = payload.get("max_tokens")

        if max_tokens:
            return int(max_tokens)

        # Dynamic defaults based on priority and daily limit
        priority = limits.get("priority", "normal")
        daily_limit = limits.get("daily_limit", 100000)

        priority_multipliers = {"low": 0.1, "normal": 0.2, "high": 0.3, "critical": 0.5}

        multiplier = priority_multipliers.get(priority, 0.2)
        default_reservation = min(int(daily_limit * multiplier), 1000)  # Cap at 1000

        logger.info(f"No max_tokens specified, using default reservation: {default_reservation}")
        return default_reservation

    async def _check_and_reserve_quota(
        self, api_key: str, reservation: int, limits: Dict[str, Any], req_priority: str
    ):
        """Check quota limits and reserve tokens"""
        used_d, used_m = await self.quota.get_usage(api_key)

        if used_d + reservation > limits["daily_limit"]:
            self.metrics.quota_rejections_total.labels(
                api_key=api_key, priority=req_priority, scope="daily"
            ).inc()
            raise HTTPException(
                status_code=429,
                detail=f"Daily quota would be exceeded. Requested: {reservation}, "
                f"Available: {limits['daily_limit'] - used_d}",
            )

        if used_m + reservation > limits["monthly_limit"]:
            self.metrics.quota_rejections_total.labels(
                api_key=api_key, priority=req_priority, scope="monthly"
            ).inc()
            raise HTTPException(
                status_code=429,
                detail=f"Monthly quota would be exceeded. Requested: {reservation}, "
                f"Available: {limits['monthly_limit'] - used_m}",
            )

        # Reserve tokens upfront
        await self.quota.reserve_tokens(api_key, reservation)
        logger.info(f"Reserved {reservation} tokens for streaming request")

    async def _validate_upstream_connectivity(self, request, full_path: str, body_bytes: bytes):
        """Validate that upstream is reachable before starting stream"""
        try:
            # Quick connectivity test
            async with self.upstream.stream(
                request.method, full_path, dict(request.headers), body_bytes
            ) as resp:
                if not (200 <= resp.status_code < 300):
                    raise StreamingError(
                        f"Upstream returned status {resp.status_code}",
                        status_code=resp.status_code,
                    )

                # Try to get first chunk to ensure streaming works
                chunk_count = 0
                async for chunk in resp.aiter_raw():
                    if chunk:
                        chunk_count += 1
                        break  # Just verify we can get first chunk

                if chunk_count == 0:
                    raise StreamingError("No data received from upstream")

        except StreamingError:
            raise
        except Exception as e:
            logger.error(f"Upstream connectivity validation failed: {e}")
            raise StreamingError(f"Failed to connect to upstream: {str(e)}")

    async def _stream_with_quota_management(
        self,
        request,
        full_path: str,
        body_bytes: bytes,
        api_key: str,
        req_priority: str,
        reservation: int,
    ) -> AsyncGenerator[bytes, None]:
        """Stream response while managing quotas and tracking usage"""

        response_body = bytearray()
        actual_tokens = 0
        chunks_sent = 0
        stream_started = False
        last_error = None

        try:
            async with self.upstream.stream(
                request.method, full_path, dict(request.headers), body_bytes
            ) as resp:

                if not (200 <= resp.status_code < 300):
                    raise StreamingError(
                        f"Upstream error: {resp.status_code}",
                        status_code=resp.status_code,
                    )

                stream_started = True

                # Stream chunks while parsing for token usage
                async for chunk in resp.aiter_raw():
                    if chunk:
                        response_body.extend(chunk)
                        chunks_sent += 1

                        # Try to parse tokens from this chunk
                        chunk_tokens = self.parser.parse_openai_chunk(chunk)
                        if chunk_tokens and chunk_tokens > actual_tokens:
                            actual_tokens = chunk_tokens

                        # Yield chunk to client
                        yield chunk

                        # Optional: Add streaming rate limiting per chunk
                        if chunks_sent % 100 == 0:  # Every 100 chunks, yield control
                            await asyncio.sleep(0.001)  # Prevent overwhelming

                logger.info(f"Streaming completed. Chunks sent: {chunks_sent}")

        except asyncio.CancelledError:
            logger.warning("Streaming request was cancelled by client")
            last_error = "cancelled"
            raise
        except StreamingError as e:
            logger.error(f"Streaming error: {e.message}")
            last_error = e.message
            if not stream_started:
                # If stream hasn't started, we can still return proper HTTP error
                raise HTTPException(status_code=e.status_code, detail=e.message)
            else:
                # Stream has started, send error as SSE
                error_chunk = f'data: {json.dumps({"error": e.message})}\n\n'
                yield error_chunk.encode()
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            last_error = str(e)
            if not stream_started:
                raise HTTPException(status_code=502, detail="Upstream service failed")
            else:
                error_data = {"error": "Service temporarily unavailable"}
                error_chunk = f"data: {json.dumps(error_data)}\n\n"
                yield error_chunk.encode()

        finally:
            # Always handle quota cleanup
            await self._handle_quota_cleanup(
                api_key,
                req_priority,
                reservation,
                actual_tokens,
                response_body,
                last_error,
            )

    async def _handle_quota_cleanup(
        self,
        api_key: str,
        req_priority: str,
        reservation: int,
        actual_tokens: int,
        response_body: bytearray,
        last_error: Optional[str],
    ):
        """Handle quota refunding and metrics after streaming completes"""

        try:
            # If we didn't get tokens from streaming chunks, try parsing full response
            if actual_tokens == 0 and response_body:
                actual_tokens = self.parser.parse_accumulated_response(bytes(response_body))

            if last_error:
                # On error, refund all reserved tokens
                if reservation > 0:
                    await self.quota.refund_tokens(api_key, reservation)
                    logger.info(f"Refunded all {reservation} tokens due to error: {last_error}")

            elif actual_tokens > 0:
                # Successful completion with usage data
                refund_amount = max(0, reservation - actual_tokens)

                if refund_amount > 0:
                    await self.quota.refund_tokens(api_key, refund_amount)
                    logger.info(
                        f"Refunded {refund_amount} tokens. Used: {actual_tokens}, "
                        f"Reserved: {reservation}"
                    )

                # Record actual usage in metrics
                self.metrics.tokens_used_total.labels(api_key=api_key, priority=req_priority).inc(
                    actual_tokens
                )

            else:
                # No usage data available, keep full reservation
                logger.warning(
                    f"No token usage data found, keeping full reservation of " f"{reservation}"
                )
                self.metrics.tokens_used_total.labels(api_key=api_key, priority=req_priority).inc(
                    reservation
                )

        except Exception as e:
            logger.error(f"Error during quota cleanup: {e}")
            # In case of cleanup error, err on the side of not charging
            if reservation > 0:
                try:
                    await self.quota.refund_tokens(api_key, reservation)
                    logger.info(
                        f"Emergency refund of {reservation} tokens due to cleanup " f"error"
                    )
                except Exception as cleanup_err:
                    logger.error(f"Failed emergency refund: {cleanup_err}")


# Updated main.py streaming section
async def handle_streaming_request_enhanced(
    request,
    full_path: str,
    api_key: str,
    limits: Dict[str, Any],
    req_priority: str,
    payload: Dict[str, Any],
    body_bytes: bytes,
    streaming_handler: StreamingHandler,
) -> StreamingResponse:
    """Enhanced streaming request handler to replace the existing one"""

    return await streaming_handler.handle_streaming_request(
        request, full_path, api_key, limits, req_priority, payload, body_bytes
    )
