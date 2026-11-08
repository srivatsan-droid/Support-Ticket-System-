"""FastAPI endpoints for streamed, cited answers and live FAQ ingestion."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from generator import ABSTENTION, extract_cited_ids, generate_support_answer
from ingest import COLLECTION_NAME, init_collection, upsert_entry
from retrieval import retrieve

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Make /ingest work on a fresh Qdrant without running ingest.py first.
    try:
        init_collection()
    except Exception:
        logger.exception("Could not initialise the Qdrant collection at startup.")
    yield


app = FastAPI(
    title="TaskFlow Support & FAQ RAG Assistant",
    version="2.1.0",
    description="Grounded support answers streamed as newline-delimited JSON events.",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="Customer support question")
    category: str | None = Field(default=None, description="Optional category filter")

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value


class QueryResponse(BaseModel):
    answer: str
    cited_sources: list[str]


class QueryStreamEvent(BaseModel):
    """NDJSON events: sources, answer delta(s), then a complete QueryResponse.

    `retrieved_sources` (first event) = chunks supplied to the model.
    `cited_sources` (final event) = IDs the model actually cited in its answer.
    """
    type: Literal["sources", "delta", "final", "error"]
    retrieved_sources: list[str] | None = None
    cited_sources: list[str] | None = None
    text: str | None = None
    answer: str | None = None
    detail: str | None = None


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    category: str = Field(min_length=1)
    last_updated: str = Field(min_length=1, description="Date in YYYY-MM-DD format")

    @field_validator("id", "title", "body", "category", "last_updated")
    @classmethod
    def nonblank_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Fields must not be blank.")
        return value

    @field_validator("last_updated")
    @classmethod
    def valid_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("last_updated must use YYYY-MM-DD format.") from exc
        return value


def _event(event: QueryStreamEvent) -> str:
    return json.dumps(event.model_dump(exclude_none=True), ensure_ascii=False) + "\n"


@app.post(
    "/query",
    summary="Ask a question (streamed NDJSON)",
    responses={
        200: {
            "description": (
                "Newline-delimited JSON stream: one `sources` event (retrieved_sources), then `delta` "
                "events (text), then a `final` event whose `answer` and `cited_sources` are "
                "the QueryResponse shown here. On failure the last event has type `error`."
            ),
            "model": QueryResponse,  # shape of the final event (answer + cited_sources)
            "content": {"application/x-ndjson": {"schema": {"$ref": "#/components/schemas/QueryResponse"}}},
        },
        503: {"description": "Retrieval service unavailable."},
    },
)
def query_endpoint(req: QueryRequest):
    try:
        retrieved = retrieve(req.question, category=req.category, top_k=3)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Retrieval service is unavailable.") from exc
    retrieved_ids = list(dict.fromkeys(item.payload["entry_id"] for item in retrieved))

    def stream():
        yield _event(QueryStreamEvent(type="sources", retrieved_sources=retrieved_ids))
        answer_parts: list[str] = []
        try:
            # Empty/low-confidence retrieval abstains without calling the LLM.
            generated = [ABSTENTION] if not retrieved else generate_support_answer(req.question, retrieved)
            for token in generated:
                answer_parts.append(token)
                yield _event(QueryStreamEvent(type="delta", text=token))
            answer = "".join(answer_parts)
            final = QueryResponse(answer=answer, cited_sources=extract_cited_ids(answer, retrieved_ids))
            yield _event(QueryStreamEvent(type="final", **final.model_dump()))
        except Exception:
            logger.exception("Answer generation failed")
            yield _event(QueryStreamEvent(
                type="error",
                detail="Answer generation failed. Please retry; no answer was produced.",
            ))

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/ingest", status_code=status.HTTP_201_CREATED)
def ingest_endpoint(entry: IngestRequest):
    try:
        chunks = upsert_entry(entry.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=503, detail="Ingestion service is unavailable.") from exc
    return {"status": "success", "entry_id": entry.id, "chunks_upserted": chunks}


@app.get("/health")
def health_check():
    return {"status": "healthy", "collection": COLLECTION_NAME}
