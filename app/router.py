import re


# ============================================================
# QUERY ROUTER
# ============================================================

GREETINGS = {
    "hi", "hii", "hiii", "hello", "helo", "helloo", "hey", "heyy", "hi there",
    "hello there", "hey there", "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening", "namaste",
}

THANKS = {
    "thanks", "thank you", "thankyou", "thx", "ty", "thanks a lot",
    "thank you so much", "many thanks", "great thanks", "ok thanks", "okay thanks",
}

GOODBYES = {
    "bye", "goodbye", "bye bye", "see you", "see you later", "talk to you later",
    "thats all", "that is all",
}

IDENTITY = {
    "who are you", "what are you", "what is your name", "whats your name",
    "are you a bot", "are you human",
}

CAPABILITY = {
    "what can you do", "what can you help with", "how can you help me",
    "what do you help with", "what information can you provide",
    "what topics can you help with", "help",
}

RESPONSES = {
    "greeting": "Hi! How can I help you with Consiva today?",
    "thanks": "You're welcome! Let me know if you need anything else about Consiva.",
    "goodbye": "Goodbye! Have a great day.",
    "identity": (
        "I'm Consiva's AI Assistant. I answer questions about Consiva and DPDP "
        "compliance using the information on consiva.ai."
    ),
    "capability": (
        "I can help with Consiva's plans and pricing, product features, policies, "
        "and DPDP Act topics such as consent, Data Principal rights, breach "
        "reporting and penalties."
    ),
}


def normalize(question: str) -> str:
    """Lower-case, drop punctuation and emoji, collapse spaces."""
    question = question.lower().replace("'", "").replace("’", "")
    question = re.sub(r"[^a-z0-9 ]+", " ", question)
    return " ".join(question.split())


def classify_query(question: str) -> str:
    """
    Classify the user's query before sending it to RAG.

    Returns one of: greeting, thanks, goodbye, identity, capability, rag
    """

    text = normalize(question)

    for query_type, phrases in (
        ("greeting", GREETINGS),
        ("thanks", THANKS),
        ("goodbye", GOODBYES),
        ("identity", IDENTITY),
        ("capability", CAPABILITY),
    ):
        if text in phrases:
            return query_type

    return "rag"


def get_direct_response(query_type: str):
    """Return a response for queries that don't need RAG."""
    return RESPONSES.get(query_type)


# The user explicitly asks for a long answer. Everything else is answered
# at the length its facts need (see FORMAT_RULES in app/llm.py).
WANTS_DETAIL = re.compile(
    r"\b(in detail|in depth|detailed|elaborate|step[- ]by[- ]step|comprehensive|deep dive|"
    r"full (list|details|explanation)|complete (list|guide|details)|everything (about|on)|"
    r"tell me more|more (detail|details|info|information)|walk me through|"
    r"explain (fully|thoroughly|everything|all))\b",
    re.IGNORECASE,
)


def wants_detail(question: str) -> bool:
    return bool(WANTS_DETAIL.search(question))


# The user asks what to do or how to do something: the answer is a
# numbered list of steps (the model alone often falls back to bullets).
WANTS_STEPS = re.compile(
    r"\bhow (do|can|should|to)\b(?! (you|they) (define|mean))|"
    r"\bwhat (should|must|do|does) .{0,40}\b(do|take)\b|"
    r"\b(steps?|step[- ]by[- ]step|process|procedure|checklist|set ?up|install|implement|configure)\b|"
    r"\bafter (a|an|the)\b",
    re.IGNORECASE,
)


def wants_steps(question: str) -> bool:
    return bool(WANTS_STEPS.search(question))
