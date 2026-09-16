
import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
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
  and Enterprise plan information when available in the context.
- If the user asks about plans, pricing, or subscriptions,
  treat the question as a pricing-related question.
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
# LOAD GEMINI
# ============================================================

def load_llm():

    if not GEMINI_API_KEY:

        raise EnvironmentError(
            "GEMINI_API_KEY not found in .env"
        )

    logger.info(
        f"Initializing Gemini: {GEMINI_MODEL}"
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    logger.info(
        "Gemini initialized successfully"
    )

    return llm


# ============================================================
# RUN MODEL
# ============================================================

def run_model(
    model,
    question: str,
    context: str
) -> str:
    """
    Execute Gemini and return a clean answer.
    """

    logger.info(
        "Starting Gemini inference..."
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
        "Gemini inference successful"
    )

    return answer


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str,
    llm=None
) -> str:
    """
    Generate an answer using Gemini only.
    """

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
    # GEMINI
    # ========================================================

    try:

        if llm is None:

            llm = load_llm()

        logger.info(
            "Generating answer with Gemini..."
        )

        return run_model(
            model=llm,
            question=question,
            context=context,
        )

    except Exception as error:

        logger.error(
            f"Gemini inference failed: {error}",
            exc_info=True,
        )

        return (
            "Unable to connect to the AI assistant "
            "right now. Please try again later."
        )
