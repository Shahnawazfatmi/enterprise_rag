from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Literal
import json
import logging
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.llm import LLMUnavailableError
from app.rag import ConsivaRAG, KnowledgeBaseUnavailableError
from config import ALLOWED_ORIGINS, MAX_QUESTION_CHARS, RATE_LIMIT_PER_MINUTE

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Consiva AI Assistant",
    description="RAG-based compliance assistant for Consiva.ai",
    version="2.0.0",
)

# Set ALLOWED_ORIGINS in .env (comma-separated) to your site's origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

rag = ConsivaRAG()


# ============================================================
# RATE LIMIT (per client IP, in memory)
# ============================================================

_requests = defaultdict(deque)
_requests_lock = threading.Lock()


def check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()

    with _requests_lock:
        window = _requests[client_ip]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many questions. Please wait a minute.")

        window.append(now)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    id: int
    title: str
    section: str = ""
    url: str = ""


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    query_type: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    status = rag.status()
    healthy = status["index_loaded"] and status["llm_configured"]

    return {
        "status": "ok" if healthy else "degraded",
        "service": "consiva-ai-assistant",
        **status,
    }


@contextmanager
def http_errors():
    """Turn RAG errors into HTTP errors (shared by /ask and /ask/stream)."""

    try:
        yield

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except LLMUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=f"{error} Please try again later.",
        )

    except KnowledgeBaseUnavailableError:
        logger.exception("Knowledge base unavailable")
        raise HTTPException(status_code=503, detail="The knowledge base is not available right now.")

    except Exception:
        # Full details go to the server log only.
        logger.exception("Unexpected error while answering")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest, request: Request):

    check_rate_limit(request.client.host if request.client else "unknown")

    with http_errors():
        return rag.ask_question(
            payload.question,
            history=[message.model_dump() for message in payload.history],
        )


def sse(event: str, data: dict) -> str:
    """One server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_question_stream(payload: QuestionRequest, request: Request):
    """
    /ask as server-sent events, so the widget can show the answer while
    the LLM writes it:

        delta {"text"}                         a piece of the answer
        reset {}                               discard the text so far (fallback LLM starts over)
        done  {"answer", "sources", "query_type"}   same fields as /ask; "answer" is final
        error {"status", "detail"}             the LLM failed after streaming started

    Bad input, rate limits and retrieval errors still return normal HTTP
    errors, because they happen before streaming starts.
    """

    check_rate_limit(request.client.host if request.client else "unknown")

    with http_errors():
        result, prepared = rag.prepare(
            payload.question,
            history=[message.model_dump() for message in payload.history],
        )

    def events():
        if result is not None:
            yield sse("delta", {"text": result["answer"]})
            yield sse("done", result)
            return

        try:
            for event, data in rag.stream_answer(prepared):
                yield sse(event, data)

        except LLMUnavailableError as error:
            yield sse("error", {"status": 503, "detail": f"{error} Please try again later."})

        except Exception:
            logger.exception("Unexpected error while streaming")
            yield sse("error", {"status": 500, "detail": "Something went wrong. Please try again."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # X-Accel-Buffering: nginx would otherwise hold the stream back until it ends.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
