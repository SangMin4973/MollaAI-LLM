import json
import logging
import time
import uuid
from typing import Any, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from llm import QwenChat
from qdrant_store import UserMemoryStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molla.llm")
chat = QwenChat()
memory_store = UserMemoryStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    memory_store.ensure_collection()
    logger.info(
        "qdrant_ready collection=%s url=%s",
        memory_store.collection_name,
        memory_store.url,
    )
    yield


app = FastAPI(title="molla-llm", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


class MemoryPointPayload(BaseModel):
    userId: str | None = None
    phoneNumber: str
    userText: str
    assistantText: str | None = None
    createdAt: str
    audioKey: str | None = None


class MemoryPoint(BaseModel):
    id: str
    vector: list[float]
    payload: MemoryPointPayload


class MemoryUpsertRequest(BaseModel):
    points: list[MemoryPoint] = Field(min_length=1)


async def read_streamed_text(request: Request) -> str:
    chunks: list[str] = []

    async for chunk in request.stream():
        if not chunk:
            continue
        chunks.append(chunk.decode("utf-8"))

    raw_text = "".join(chunks).strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty streamed body")

    if "data:" in raw_text:
        sse_chunks: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            sse_chunks.append(payload)
        if sse_chunks:
            raw_text = " ".join(sse_chunks).strip()

    ndjson_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if ndjson_lines and all(line.startswith("{") and line.endswith("}") for line in ndjson_lines):
        text_parts: list[str] = []
        for line in ndjson_lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                text_parts = []
                break

            if isinstance(data, dict):
                text = data.get("text") or data.get("query") or data.get("partial")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())

        if text_parts:
            raw_text = " ".join(text_parts).strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="No text extracted from stream")

    return raw_text


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    return ChatResponse(answer=chat.ask(payload.query))


def token_event_stream(query: str, request_id: str) -> Iterator[str]:
    started_at = time.perf_counter()
    logger.info(
        "chat_request_received request_id=%s query_len=%s",
        request_id,
        len(query),
    )
    token_count = 0
    for token in chat.stream_answer(query, request_id=request_id):
        token_count += 1
        payload = json.dumps({"token": token}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
    logger.info(
        "chat_response_done request_id=%s elapsed_ms=%s tokens=%s",
        request_id,
        int((time.perf_counter() - started_at) * 1000),
        token_count,
    )
    yield "data: [DONE]\n\n"


@app.post("/chat/tokens")
def chat_token_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    request_id = uuid.uuid4().hex[:12]
    logger.info("chat_stream_open request_id=%s", request_id)
    return StreamingResponse(token_event_stream(payload.query, request_id), media_type="text/event-stream")


@app.post("/chat/stream", response_model=ChatResponse)
async def chat_stream_endpoint(request: Request) -> ChatResponse:
    query = await read_streamed_text(request)
    return ChatResponse(answer=chat.ask(query))


@app.api_route("/memory/points", methods=["POST", "PUT"])
def upsert_memory_points(payload: MemoryUpsertRequest) -> dict[str, Any]:
    points = [point.model_dump(mode="python") for point in payload.points]
    memory_store.upsert_points(points)
    logger.info("memory_points_upserted count=%s", len(points))
    return {"status": "ok", "count": len(points)}
