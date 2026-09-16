from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.rag import ConsivaRAG


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Consiva AI Assistant",
    description="RAG-based compliance assistant for Consiva.ai",
    version="1.0.0",
)


# ============================================================
# CORS
# Allows the React frontend to communicate with FastAPI
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# LOAD RAG
# ============================================================

rag = ConsivaRAG()


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "consiva-ai-assistant"
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        result = rag.ask_question(request.question)

        return {
            "answer": result["answer"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )