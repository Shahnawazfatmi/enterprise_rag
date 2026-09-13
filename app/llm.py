
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

LLM_BASE_URL = "http://172.16.10.117:8080/v1"

LLM_MODEL = (
    "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"
)

LLM_API_KEY = "not-needed"


# ============================================================
# GEMINI FALLBACK CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Consiva.ai's compliance assistant.

Answer the user's question using ONLY the provided context.

Rules:

- Give a direct, natural, conversational answer.
- Do not copy the context unnecessarily.
- Synthesize information from multiple relevant chunks when needed.
- Include important numbers, prices, dates, limits, and legal sections when relevant.
- Do not invent or assume information.
- Do not use outside knowledge.
- Evaluate the response based ONLY on the provided Source Text.
- Do not use your pretrained knowledge to validate, expand, or add context.
- If the answer cannot be found in the provided context, say:
  "I couldn't find that information in the knowledge base."
- Keep the answer concise while including necessary information.
- For pricing questions, provide the relevant Free, Pro,
- most important lemme know about plans means user asking for prising, so searched for prising related document.
  and Enterprise plan information when available in the context.
- Use bullet points only when they improve readability.

Source Text:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# ============================================================
# RESPONSE CONTENT EXTRACTION
# ============================================================

def extract_response_content(response) -> str:
    """
    Convert LangChain model response content into a clean string.
    """

    content = getattr(response, "content", None)

    if content is None:
        raise RuntimeError(
            "LLM returned no response content."
        )

    if isinstance(content, str):

        answer = content

    elif isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

        answer = "".join(parts)

    else:

        answer = str(content)

    answer = answer.strip()

    if not answer:

        raise RuntimeError(
            "LLM returned an empty response."
        )

    return answer


# ============================================================
# LOAD GEMMA
# ============================================================

def load_llm():

    logger.info(
        f"Initializing local Gemma client: {LLM_MODEL}"
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.2,
        max_tokens=1024,
        timeout=30,
        max_retries=0,
    )

    logger.info(
        "Gemma client initialized successfully"
    )

    return llm


# ============================================================
# LOAD GEMINI FALLBACK
# ============================================================

def load_gemini():

    if not GEMINI_API_KEY:

        raise EnvironmentError(
            "GEMINI_API_KEY not found in .env"
        )

    logger.info(
        f"Initializing Gemini fallback: {GEMINI_MODEL}"
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    logger.info(
        "Gemini fallback client initialized successfully"
    )

    return llm


# ============================================================
# RUN MODEL
# ============================================================

def run_model(
    model,
    model_name: str,
    question: str,
    context: str
) -> str:
    """
    Execute the LLM and return a clean answer.
    Any inference failure is propagated to the caller.
    """

    logger.info(
        f"Starting inference with {model_name}..."
    )

    chain = PROMPT | model

    response = chain.invoke(
        {
            "question": question.strip(),
            "context": context,
        }
    )

    answer = extract_response_content(response)

    logger.info(
        f"{model_name} inference successful"
    )

    return answer


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
    llm=None
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not context or not context.strip():

        return (
            "I couldn't find that information "
            "in the knowledge base."
        )

    question = question.strip()

    # ========================================================
    # 1. TRY GEMMA
    # ========================================================

    try:

        if llm is None:

            llm = load_llm()

        logger.info(
            "Attempting answer generation with Gemma..."
        )

        return run_model(
            model=llm,
            model_name="Gemma",
            question=question,
            context=context,
        )

    except Exception as gemma_error:

        logger.error(
            f"Gemma inference failed: {gemma_error}",
            exc_info=True,
        )

        logger.warning(
            "Gemma unavailable. "
            "Switching to Gemini fallback..."
        )

    # ========================================================
    # 2. TRY GEMINI FALLBACK
    # ========================================================

    try:

        gemini = load_gemini()

        logger.info(
            "Attempting answer generation with Gemini..."
        )

        return run_model(
            model=gemini,
            model_name="Gemini",
            question=question,
            context=context,
        )

    except Exception as gemini_error:

        logger.error(
            f"Gemini fallback failed: {gemini_error}",
            exc_info=True,
        )

        # ----------------------------------------------------
        # BOTH MODELS FAILED
        # ----------------------------------------------------

        return (
            "Unable to connect to the AI assistant "
            "right now. Please try again later."
        )

